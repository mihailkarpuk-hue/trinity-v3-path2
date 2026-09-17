#!/usr/bin/env python3
"""Тринити V3 · позвоночник — локальный HTTP-сервер на порту 8004.

Запускается из ЗАПУСТИТЬ.command. Раздаёт статику из корня
«Тринити V3 · позвоночник/». Cache-Control: no-store, чтобы изменения
сразу подхватывались в браузере.
"""
import http.server
import socketserver
import os

PORT = 8004
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


# Python на macOS отдаёт .m4a как audio/mp4a-latm — браузер такой тип НЕ играет.
# Чиним на audio/mp4 (буквица_живая звучит).
Handler.extensions_map['.m4a'] = 'audio/mp4'


socketserver.TCPServer.allow_reuse_address = True   # перезапуск без «Address already in use»
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Тринити V3 · позвоночник запущена: http://localhost:{PORT}/")
    httpd.serve_forever()
