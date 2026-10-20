import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler,HTTPServer
from maze.backends.chat_api import ChatAPI


class LocalHTTPTests(unittest.TestCase):
    def test_payload_and_usage_roundtrip_without_external_model(self):
        seen=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append(payload)
                output={'model':'mock-1','choices':[{'message':{'content':'Final answer: A'},'finish_reason':'stop'}],
                        'usage':{'prompt_tokens':17,'completion_tokens':4}}
                body=json.dumps(output).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def log_message(self,*args):pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            target=ChatAPI(f'http://127.0.0.1:{server.server_port}/v1',key_env='UNSET_TEST_KEY')
            output=target.generate({'model':'mock-1','seed':12,'messages':[{'role':'user','content':'Invented question'}],
                                    'decoding':{'temperature':.3,'max_tokens':20,'chat_template_kwargs':{'enable_thinking':False}}})
            self.assertEqual((output.input_tokens,output.output_tokens),(17,4))
            self.assertEqual(seen[0]['n'],1);self.assertEqual(seen[0]['seed'],12)
            self.assertFalse(seen[0]['chat_template_kwargs']['enable_thinking'])
            self.assertNotIn('answer',seen[0])
        finally:server.shutdown();server.server_close();worker.join()
