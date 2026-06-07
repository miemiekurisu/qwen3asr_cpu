#!/usr/bin/env python3
"""
HTTPS reverse proxy for qasr_server (HTTP backend).

浏览器 mic 权限要求 secure context (https://localhost 或 https://<lan-ip>),
qasr_server 本身是明文 HTTP, 所以在前面套一层自签 TLS 反代.

用法:
    python3 tools/https_proxy.py                           # 0.0.0.0:19992 -> 127.0.0.1:19991, 每次启动生成新 cert
    python3 tools/https_proxy.py --bind-port 443           # 改监听
    python3 tools/https_proxy.py --upstream-port 19991     # 改后端
    python3 tools/https_proxy.py --reuse-cert              # 复用 --cert-dir 下的 cert (默认每次新建)
    python3 tools/https_proxy.py --cert-dir /path/dir      # 指定 cert 目录 (默认 mktemp -d, 退出时删)
    python3 tools/https_proxy.py --cert a.pem --key b.pem  # 完全指定路径 (与 --reuse-cert 一起用)
    python3 tools/https_proxy.py --generate-cert           # 单独生成 cert, 不启动代理

依赖:
    Python 3.7+ 标准库 (http.server, ssl, urllib). 无第三方依赖.
    openssl 用于 cert 生成 (apt install openssl / brew install openssl)
    无 openssl 时回退到 cryptography 库 (pip install cryptography)

安全 / 卫生:
    - 默认每次启动生成新 cert (自签, 浏览器每次都要"高级→继续"一次)
    - cert 写在 mktemp -d 创建的临时目录, Ctrl+C / SIGTERM 时自动删
    - 想跨进程复用 cert: --reuse-cert --cert-dir <持久目录>
    - 想完全自管: --cert a.pem --key b.pem (用你自己的 CA 签的)

浏览器首次访问会警告"证书无效", 选"高级"→"继续访问"即可.
LAN 上要全设备信任, 把 cert 装到系统/浏览器 trust store — 涉及持久 cert, 不在本工具范围.

HTTPS 方案对比:
    A. qasr_server 自带 TLS (mbedTLS 静态链接, 跨平台) — 待集成, 完成后推荐
    B. 本工具 (Python 反代)                              — 当前默认, Linux 自带 py3, Windows 需装
    C. Caddy / nginx 反代                                 — 需系统装, 适合生产

对 (A) 集成完成前, 用 (B) 过渡.
"""
import argparse
import atexit
import http.client
import http.server
import ipaddress
import os
import shutil
import signal
import socketserver
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

DEFAULT_UPSTREAM = "http://127.0.0.1:19991"
DEFAULT_BIND = ("0.0.0.0", 19992)


def generate_self_signed_cert(cert_path: str, key_path: str,
                              sans: list = None) -> None:
    """用 openssl 生成自签 cert. openssl 不在则回退到 cryptography."""
    sans = sans or ["DNS:localhost", "IP:127.0.0.1", "IP:::1"]

    cmd = [
        "openssl", "req", "-x509", "-nodes", "-newkey", "rsa:2048",
        "-keyout", key_path,
        "-out",    cert_path,
        "-days",   "365",
        "-subj",   "/CN=qasr-server (self-signed, ephemeral)",
        "-addext", "subjectAltName=" + ",".join(sans),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        if isinstance(e, FileNotFoundError):
            print("[proxy] openssl 不在, 回退到 cryptography", file=sys.stderr)
        else:
            print(f"[proxy] openssl 失败 ({e}), 回退到 cryptography", file=sys.stderr)
        _generate_via_cryptography(cert_path, key_path, sans)


def _generate_via_cryptography(cert_path, key_path, sans):
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
    except ImportError:
        print("[proxy] ERROR: openssl 不在, 且 cryptography 库未装.", file=sys.stderr)
        print("        解决: apt install openssl   或   pip install cryptography",
              file=sys.stderr)
        sys.exit(1)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "qasr-server (self-signed, ephemeral)")
    ])
    san_list = []
    for s in sans:
        if s.startswith("DNS:"):
            san_list.append(x509.DNSName(s[4:]))
        elif s.startswith("IP:"):
            san_list.append(x509.IPAddress(ipaddress.ip_address(s[3:])))
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .sign(key, hashes.SHA256())
    )
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """HTTP/1.1 透明反代. 任何 method 透传到 upstream.

    Streaming-aware: the upstream may send
    ``Content-Type: text/event-stream`` (SSE) where the response
    is delivered as the server emits frames, with no
    ``Content-Length`` and no ``Transfer-Encoding: chunked`` —
    just ``Connection: close`` to signal end-of-stream.  In
    that case ``urllib.request.urlopen`` would buffer the whole
    body until EOF, defeating SSE entirely.  We detect this
    pattern and fall back to a raw ``http.client`` connection
    that streams chunks as they arrive.  Buffered responses
    still go through the urllib path. """
    protocol_version = "HTTP/1.1"
    upstream: str = DEFAULT_UPSTREAM  # 类变量, main() 注入

    def _flush_chunk(self, data: bytes) -> None:
        """Write one HTTP chunk to the client."""
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _proxy_buffered(self, method: str) -> None:
        """Path for buffered responses (everything except
        streaming SSE).  Uses urllib for simplicity. """
        url = self.upstream + self.path
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        forward_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "connection", "transfer-encoding")
        }
        req = urllib.request.Request(url, data=body, method=method, headers=forward_headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp_body = r.read()
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read() if e.fp else b"")

    def _proxy_streaming(self, method: str) -> None:
        """Path for SSE: open a raw http.client connection
        and stream chunks to the client as they arrive.  We
        set ``Transfer-Encoding: chunked`` on the client side
        so each upstream chunk becomes one HTTP chunk, which
        is what the browser's EventSource expects. """
        from urllib.parse import urlparse
        u = urlparse(self.upstream)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        # Build the request line + headers
        path = u.path.rstrip("/") + self.path
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        forward_headers = []
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "connection", "transfer-encoding"):
                continue
            forward_headers.append((k, v))
        header_block = f"{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
        header_block += "".join(f"{k}: {v}\r\n" for k, v in forward_headers)
        if body is not None:
            header_block += f"Content-Length: {len(body)}\r\n"
        header_block += "Connection: close\r\n\r\n"
        try:
            conn = http.client.HTTPConnection(host, port, timeout=300)
            # request() handles header serialization + body
            # sending.  We pass the upstream's Host header
            # because the client sent a Host like
            # "127.0.0.1:19992" but the upstream is on
            # "127.0.0.1:19991" — cpp-httplib may use Host for
            # virtual hosting, so we strip the client's Host
            # and let request() set the upstream one.
            upstream_headers = dict(forward_headers)
            upstream_headers.pop("Host", None)
            conn.request(method, path, body=body, headers=upstream_headers)
            resp = conn.getresponse()
            # Send status + headers to client.  Force
            # chunked so the browser flushes per frame.
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in ("transfer-encoding", "connection", "content-length"):
                    continue
                self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # 1-byte reads are required: when the upstream
            # uses Connection: close with no Content-Length
            # and no chunked (qasr_server's SSE style),
            # http.client.HTTPResponse.read(amt) blocks
            # until EOF for any `amt` larger than what
            # the BufferedReader has buffered.  The 1-byte
            # read goes straight to the underlying socket,
            # surfacing partial data to us as soon as the
            # upstream writes.  We accumulate into a buffer
            # and flush at SSE frame boundaries ("\n\n") so
            # the browser sees each event with sub-frame
            # latency.  Larger amt (e.g. 8 KB) would buffer
            # up to 8 KB before surfacing, defeating SSE.
            acc = bytearray()
            while True:
                try:
                    b = resp.read(1)
                except Exception:
                    break
                if not b:
                    break
                acc.extend(b)
                if acc.endswith(b"\n\n") or len(acc) >= 4096:
                    self._flush_chunk(bytes(acc))
                    acc.clear()
            if acc:
                self._flush_chunk(bytes(acc))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception as e:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass

    def _proxy(self, method: str) -> None:
        """Peek the upstream's response code & Content-Type
        cheaply using a HEAD-ish probe, then dispatch.  The
        probe costs one extra request for non-streaming paths
        (which is fine — they're sub-millisecond locally); for
        streaming paths it saves us from buffering SSE.  We
        instead use a simple heuristic: try the buffered path
        first only for non-GET.  For GET, always go streaming
        — the upstream may be SSE and we cannot tell without
        sending the request.  The streaming path is
        functionally identical to the buffered one for
        short responses (single chunk). """
        # Simple rule: SSE is always GET.  POST/PUT/DELETE
        # never stream.  This avoids a second round-trip for
        # non-SSE cases.
        if method == "GET":
            self._proxy_streaming(method)
        else:
            self._proxy_buffered(method)

    def do_GET(self): self._proxy("GET")
    def do_POST(self): self._proxy("POST")
    def do_PUT(self): self._proxy("PUT")
    def do_DELETE(self): self._proxy("DELETE")
    def do_PATCH(self): self._proxy("PATCH")
    def do_OPTIONS(self): self._proxy("OPTIONS")
    def do_HEAD(self): self._proxy("HEAD")

    def log_message(self, fmt, *args):
        sys.stderr.write("[proxy] %s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind-host",  default=DEFAULT_BIND[0],  help="监听 host (默认 0.0.0.0)")
    ap.add_argument("--bind-port",  type=int, default=DEFAULT_BIND[1], help="监听 port (默认 19992)")
    ap.add_argument("--upstream",   default=DEFAULT_UPSTREAM,  help=f"后端 URL (默认 {DEFAULT_UPSTREAM})")
    ap.add_argument("--cert-dir",   default=None,
                    help="cert 目录 (默认 mktemp -d, 退出时自动删). --reuse-cert 时持久化")
    ap.add_argument("--cert",       default=None, help="显式指定 cert 路径 (覆盖默认)")
    ap.add_argument("--key",        default=None, help="显式指定 key  路径 (覆盖默认)")
    ap.add_argument("--reuse-cert", action="store_true",
                    help="复用 --cert-dir 下的 cert, 不再每次新建 (默认行为是每次启动新 cert)")
    ap.add_argument("--generate-cert", action="store_true", help="只生成 cert, 不启动代理")
    args = ap.parse_args()

    # ─────────── 决定 cert/key 路径 ───────────
    ephemeral = False
    if args.cert and args.key:
        # 用户完全自管
        cert_path, key_path = args.cert, args.key
    else:
        # 默认: 每次启动 mktemp -d, ephemeral
        if args.cert_dir:
            cert_dir = args.cert_dir
            os.makedirs(cert_dir, exist_ok=True)
        else:
            cert_dir = tempfile.mkdtemp(prefix="qasr_https_")
            ephemeral = True
        cert_path = args.cert or os.path.join(cert_dir, "cert.pem")
        key_path  = args.key  or os.path.join(cert_dir, "key.pem")

    if ephemeral:
        print(f"[proxy] ephemeral cert dir: {cert_dir} (退出时自动删)")
        def _cleanup():
            if os.path.isdir(cert_dir):
                shutil.rmtree(cert_dir, ignore_errors=True)
                print(f"[proxy] 删 {cert_dir}")
        atexit.register(_cleanup)
        # 兜底: SIGTERM 也能触发 atexit
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    else:
        print(f"[proxy] cert dir: {cert_dir} (持久)")

    # ─────────── 生成 / 复用 cert ───────────
    if args.reuse_cert and os.path.exists(cert_path) and os.path.exists(key_path):
        print(f"[proxy] 复用现有 cert: {cert_path}")
    else:
        if os.path.exists(cert_path):
            os.unlink(cert_path)
        if os.path.exists(key_path):
            os.unlink(key_path)
        generate_self_signed_cert(cert_path, key_path)
        print(f"[proxy] 新 cert: {cert_path}")
    if args.generate_cert:
        return

    # ─────────── 启代理 ───────────
    ProxyHandler.upstream = args.upstream
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)
    bind = (args.bind_host, args.bind_port)

    class ReusableTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(bind, ProxyHandler) as httpd:
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print(f"[proxy] HTTPS {bind[0]}:{bind[1]} -> {args.upstream}", flush=True)
        print(f"[proxy] cert={cert_path}", flush=True)
        if ephemeral:
            print(f"[proxy] cert 临时, 重启会失效. 想持久: --reuse-cert --cert-dir <dir>", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[proxy] shutdown")


if __name__ == "__main__":
    main()
