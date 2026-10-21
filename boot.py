"""Pinned checkpoint download, integrity verification, serving and API acceptance."""
import concurrent.futures
import base64
import io
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request

REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
MODEL = Path(os.environ.get('MODEL_PATH','/models/DeepSeek-V4.1-Flash'))
STATE = Path(os.environ.get('STATE_PATH','/state'))
PORT = int(os.environ.get('SERVER_PORT','8010'))

def save(name, value):
    STATE.mkdir(parents=True,exist_ok=True)
    temporary = STATE/(name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(STATE/name)

def prepare():
    from huggingface_hub import snapshot_download
    MODEL.mkdir(parents=True,exist_ok=True)
    metadata = json.load(urllib.request.urlopen(
        f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true',timeout=60))
    assert metadata['sha'] == REVISION
    snapshot_download(REPO,revision=REVISION,local_dir=str(MODEL),max_workers=4)
    receipt = STATE/'verification.json'
    old = json.loads(receipt.read_text()) if receipt.exists() else {}
    cached = {f['name']:f for f in old.get('files',[])} if old.get('revision') == REVISION else {}
    def verify(entry):
        name = entry['rfilename']
        path = MODEL/name
        assert path.resolve().is_relative_to(MODEL.resolve())
        before = path.stat()
        identity = {'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'inode':before.st_ino}
        assert before.st_size == entry['size'], f'Wrong file size: {name}'
        expected = entry['lfs']['sha256'] if entry.get('lfs') else entry['blobId']
        previous = cached.get(name,{})
        if previous.get('digest') == expected and all(previous.get(k)==v for k,v in identity.items()):
            return previous
        digest = hashlib.sha256() if entry.get('lfs') else hashlib.sha1()
        if not entry.get('lfs'):
            digest.update(f'blob {before.st_size}\0'.encode())
        with path.open('rb') as handle:
            while block := handle.read(8*1024*1024):
                digest.update(block)
        assert digest.hexdigest() == expected, f'Hash mismatch: {name}'
        after = path.stat()
        assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
        print('Verified',name,flush=True)
        return {'name':name,'digest':expected,**identity}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        files = list(pool.map(verify,metadata['siblings']))
    verified = {f['name'] for f in files}
    index = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    assert set(index.values()) <= verified
    save('verification.json',{'revision':REVISION,'status':'verified','files':files})

def key():
    STATE.mkdir(parents=True,exist_ok=True)
    path = STATE/'api-key'
    value = os.environ.get('API_KEY','').strip()
    if value:
        path.write_text(value)
        path.chmod(0o600)
    elif not path.exists():
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as handle:
            handle.write(secrets.token_urlsafe(32))
    return path.read_text().strip()

def request(path,payload=None,timeout=10):
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}'+path,
        headers={'Authorization':'Bearer '+key(),'Content-Type':'application/json'},
        data=None if payload is None else json.dumps(payload).encode())
    with urllib.request.urlopen(req,timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else {'status':response.status}

def smoke():
    result = request('/v1/chat/completions',dict(model='deepseek-v4.1-flash',temperature=0,
        chat_template_kwargs={'thinking':False},messages=[dict(role='user',
        content='What is 19 + 23? Reply only with the number.')]),timeout=300)
    choice = result['choices'][0]
    assert choice['message']['content'].strip() == '42' and choice['finish_reason'] == 'stop', result
    long_ctx = ('Context note: verify SM120 sparse-MLA prefill across the 64-token kernel boundary. ' * 6).strip()
    long_res = request('/v1/chat/completions',dict(model='deepseek-v4.1-flash',temperature=0,
        chat_template_kwargs={'thinking':False},messages=[dict(role='user',
        content=f'{long_ctx}\n\nIgnore the note above. What is 19 + 23? Reply only with the number.')]),timeout=300)
    assert long_res['usage']['prompt_tokens'] > 64 and long_res['choices'][0]['message']['content'].strip() == '42', long_res
    save('smoke.json',{'short':result,'long_prefill':long_res})
    print('Fresh inference passed: 19 + 23 = 42 (short and >64-token prefill)',flush=True)
    common = dict(model='deepseek-v4.1-flash',temperature=0,
                  chat_template_kwargs={'thinking':False})
    structured = request('/v1/chat/completions',dict(common,
        messages=[dict(role='user',content='Return an object whose answer is the integer 42.')],
        response_format={'type':'json_schema','json_schema':{'name':'answer','strict':True,
            'schema':{'type':'object','properties':{'answer':{'type':'integer'}},
                      'required':['answer'],'additionalProperties':False}}}),timeout=300)
    assert json.loads(structured['choices'][0]['message']['content']) == {'answer':42}
    save('smoke-structured.json',structured)
    tool = {'type':'function','function':{'name':'lookup_fixture',
        'description':'Retrieve a stored test value.',
        'parameters':{'type':'object','properties':{'key':{'type':'string'}},
                      'required':['key'],'additionalProperties':False}}}
    messages = [dict(role='user',content='Use lookup_fixture to retrieve the value for key alpha. Do not guess.')]
    called = request('/v1/chat/completions',dict(common,messages=messages,tools=[tool]),timeout=300)
    assistant = called['choices'][0]['message']
    calls = assistant.get('tool_calls') or []
    assert len(calls)==1 and calls[0]['function']['name']=='lookup_fixture', called
    assert json.loads(calls[0]['function']['arguments']) == {'key':'alpha'}
    messages += [assistant,dict(role='tool',tool_call_id=calls[0]['id'],content='{"value":42}')]
    continued = request('/v1/chat/completions',dict(common,messages=messages,tools=[tool]),timeout=300)
    assert '42' in continued['choices'][0]['message']['content'], continued
    save('smoke-tools.json',{'call':called,'continuation':continued})
    from PIL import Image,ImageDraw
    image = Image.new('RGB',(3024,588),'white')
    draw = ImageDraw.Draw(image)
    draw.ellipse((200,100,588,488),fill='red')
    draw.rectangle((2400,100,2788,488),fill='blue')
    buffer = io.BytesIO()
    image.save(buffer,format='PNG')
    visual = request('/v1/chat/completions',dict(common,messages=[dict(role='user',content=[
        {'type':'text','text':'Describe the two colored shapes and their left-to-right order. Be concise.'},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()}}
    ])]),timeout=300)
    text = visual['choices'][0]['message']['content'].lower()
    assert all(word in text for word in ('red','circle','blue','square')), visual
    assert visual['usage']['prompt_tokens_details']['image_tokens'] == 1024, visual
    save('smoke-vision.json',visual)
    print('Structured output, tool round trip and full-budget native vision passed.',flush=True)

def serve():
    mode = os.environ.get('OFFLOAD_MODE','nvme')
    assert mode in ('nvme','ram'), 'OFFLOAD_MODE must be nvme or ram'
    context = int(os.environ.get('CONTEXT_LENGTH','409600'))
    chunk = int(os.environ.get('CHUNKED_PREFILL_SIZE','2048'))
    assert 128 <= chunk <= context and chunk % 128 == 0, 'CHUNKED_PREFILL_SIZE must be a multiple of 128 within context'
    concurrency = int(os.environ.get('MAX_RUNNING_REQUESTS','8'))
    assert 1 <= concurrency <= 32, 'MAX_RUNNING_REQUESTS must be between 1 and 32'
    memory_fraction = float(os.environ.get('MEMORY_FRACTION','0.85'))
    assert 0.5 <= memory_fraction <= 0.95, 'MEMORY_FRACTION must be between 0.5 and 0.95'
    assert 400000 <= context <= 1048576, 'Context must be 400k through the model limit 1048576'
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total',
        '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert len(gpu) == 4 and all('RTX PRO 6000' in g and int(g.rsplit(',',1)[1]) >= 95000 for g in gpu), gpu
    if mode == 'ram':
        index = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
        shards = {index[f'layers.{layer}.engram.embed.weight'] for layer in (1,14)}
        needed = sum((MODEL/name).stat().st_size for name in shards)+24*2**30
        mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(mem['MemAvailable'].split()[0])*1024
        assert available >= needed, f'RAM offload needs at least {needed/2**30:.1f} GiB available; found {available/2**30:.1f}. Use nvme mode on a 128 GiB host.'
    args = ['--model-path',str(MODEL),'--served-model-name','deepseek-v4.1-flash',
        '--trust-remote-code','--load-format','safetensors','--tp','4','--ep-size','4',
        '--attention-backend','dsv4','--moe-runner-backend','flashinfer_mxfp4',
        '--mem-fraction-static',str(memory_fraction),'--chunked-prefill-size',str(chunk),
        '--context-length',str(context),'--max-running-requests',str(concurrency),'--cuda-graph-max-bs-decode',str(concurrency),
        '--min-free-slots-delay','1','--random-seed','0','--speculative-algorithm','DSPARK','--speculative-dspark-block-size','5',
        '--enable-decoder-swa-bounded-replay','--tool-call-parser','deepseekv41',
        '--reasoning-parser','deepseek-v41','--host','0.0.0.0','--port',str(PORT)]
    if os.environ.get('MAX_TOTAL_TOKENS'):
        total_tokens = int(os.environ['MAX_TOTAL_TOKENS'])
        assert total_tokens >= context, 'MAX_TOTAL_TOKENS must cover CONTEXT_LENGTH'
        args.extend(['--max-total-tokens',str(total_tokens)])
    save('launch.json',{'revision':REVISION,'offload_mode':mode,'args':args,'gpus':gpu})
    secret = key()
    env = dict(os.environ,DSV41_SOURCE=str(MODEL))
    process = subprocess.Popen([sys.executable,'-m','sglang.launch_server',*args,'--api-key',secret],
        env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    def stop(*_):
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    def logs():
        for line in process.stdout:
            print(line.replace(secret,'[REDACTED]'),end='',flush=True)
    threading.Thread(target=logs,daemon=True).start()
    try:
        deadline = time.monotonic()+1800
        while process.poll() is None and time.monotonic()<deadline:
            try:
                request('/health',timeout=3)
                break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError('Server failed to become healthy; inspect container logs')
        smoke()
        print(f'Ready: authenticated API on port {PORT}. Key is in /state/api-key.',flush=True)
        return process.wait()
    finally:
        stop()

if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv)>1 else 'run'
    if command == 'health': request('/health')
    elif command == 'smoke': smoke()
    elif command == 'prepare': prepare()
    elif command == 'run':
        prepare()
        sys.exit(serve())
    else: raise SystemExit('Commands: run, prepare, health, smoke')
