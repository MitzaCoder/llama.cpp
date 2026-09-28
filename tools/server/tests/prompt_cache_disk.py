#!/usr/bin/env python3
"""The prompt cache's disk tier (--cache-dir) end to end, against a real model: starts llama-server
with a small --cache-ram so that conversations leave RAM, then checks that
  1. a long conversation that others pushed out of RAM comes back from disk, far sooner than
     computing it again, and decodes the same text (greedy);
  2. its next turn goes on from it (the checkpoints kept with it roll back SWA / recurrent state
     past a re-rendered reply);
  3. all of it survives a restart (SIGTERM stores the slot and the RAM entries).

  prompt_cache_disk.py LLAMA_SERVER -- MODEL ARGS...   (e.g. -m model.gguf -ngl 999 -fa on -c 65536)
Environment: PORT (default 18282), CORPUS (a long text file), DOC_CHARS / OTHER_CHARS (how much of it
the long conversation and the others take; 120000 / 60000), CACHE_PARENT (where the temporary
--cache-dir goes; on btrfs, without compression, or reads are bound by decompression)."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

if '--' not in sys.argv or sys.argv.index('--') < 2:
    sys.exit(__doc__)
BIN = sys.argv[1]
MODEL_ARGS = sys.argv[sys.argv.index('--') + 1:]
PORT = int(os.environ.get('PORT', '18282'))
URL = f'http://127.0.0.1:{PORT}'
CACHE_DIR = tempfile.mkdtemp(prefix='llama-prompt-cache-', dir=os.environ.get('CACHE_PARENT'))
LOG = open(os.path.join(CACHE_DIR, 'server.log'), 'w')
CORPUS = os.environ.get('CORPUS', '/home/mihai/Projects/Personal/qwen-flash-engine/tests/corpus/en_book.txt')
DOC_CHARS = int(os.environ.get('DOC_CHARS', '120000'))
OTHER_CHARS = int(os.environ.get('OTHER_CHARS', '60000'))


def start():
    p = subprocess.Popen([BIN, '--port', str(PORT), '-np', '1', '--cache-ram', '1024', '--cache-dir', CACHE_DIR,
                          '--cache-dir-max', '30000', '--no-webui', '-lv', '3'] + MODEL_ARGS, stdout=LOG, stderr=LOG)
    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            sys.exit(f'server exited with {p.returncode}, see {LOG.name}')
        try:
            with urllib.request.urlopen(URL + '/health', timeout=2) as r:
                if r.status == 200:
                    print(f'server up in {time.time() - t0:.0f} s')
                    return p
        except Exception:
            pass
        time.sleep(2)
    sys.exit('server did not come up')


def stop(p):
    t0 = time.time()
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=300)
    print(f'server stopped in {time.time() - t0:.1f} s')


def chat(tag, messages, max_tokens=48):
    body = {'messages': messages, 'max_tokens': max_tokens, 'temperature': 0, 'seed': 7}
    req = urllib.request.Request(URL + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        res = json.load(r)
    wall = time.time() - t0
    t, m = res['timings'], res['choices'][0]['message']
    print(f"{tag:22s} prompt {t['prompt_n'] + t['cache_n']:6d} ({t['cache_n']:6d} cached) {t['prompt_ms'] / 1e3:7.2f} s | "
          f"wall {wall:6.2f} s | {text(res)[-60:]!r}")
    return res


def text(res):
    m = res['choices'][0]['message']
    return (m.get('reasoning_content') or '') + '|' + (m.get('content') or '')


def user(text):
    return {'role': 'user', 'content': text}


def assistant(res):
    return {'role': 'assistant', 'content': res['choices'][0]['message']['content']}


failed = False


def expect(cond, what):
    global failed
    print(('  ok   ' if cond else '  FAIL ') + what)
    failed = failed or not cond


book = open(CORPUS).read()
A1 = [{'role': 'system', 'content': 'You answer questions about the text the user gives. Be brief.'},
      user(book[:DOC_CHARS] + '\n\nWho is Charlotte Lucas going to marry? One sentence.')]

p = start()
try:
    r1 = chat('A cold', A1)
    cold = r1['timings']['prompt_ms'] / 1e3
    n1 = r1['usage']['prompt_tokens']
    # others push A out of the 1 GiB RAM cache
    chat('B', [user(book[200000:200000 + OTHER_CHARS] + '\n\nSummarize this in one sentence.')])
    chat('C', [user(book[300000:300000 + OTHER_CHARS] + '\n\nSummarize this in one sentence.')])
    r1b = chat('A again', A1)
    t = r1b['timings']
    expect(t['cache_n'] >= n1 - 8, 'A came back from disk')
    expect(t['prompt_ms'] / 1e3 < cold / 5, f"in {t['prompt_ms'] / 1e3:.2f} s against {cold:.1f} s computing it")
    expect(len(text(r1)) > 20 and text(r1b) == text(r1), 'and decodes the same text')
    A2 = A1 + [assistant(r1), user('Why does she accept him? One sentence.')]
    r2 = chat('A turn 2', A2)
    expect(r2['timings']['cache_n'] >= n1 - 8, 'the next turn goes on from it')
    chat('B again', [user(book[200000:200000 + OTHER_CHARS] + '\n\nName its main character.')])
finally:
    stop(p)

p = start()
try:
    r2b = chat('A turn 2, restarted', A2)
    t = r2b['timings']
    expect(t['cache_n'] >= r2['usage']['prompt_tokens'] - 8, 'A survives a restart')
    expect(len(text(r2)) > 20 and text(r2b) == text(r2), 'and decodes the same text')
finally:
    stop(p)

files = [f for f in os.listdir(os.path.join(CACHE_DIR, [d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))][0]))]
print(f'disk: {files}')
print(f'log: {LOG.name}')
print('FAIL' if failed else 'OK')
sys.exit(1 if failed else 0)
