# dub_engines

Engines de IA em Python, cada uma rodando como processo separado controlado
pelo Rust via JSON-lines (ver `dub_engines/protocol.py`).

Teste manual de uma engine:

    python -m dub_engines mock
    {"id":1,"method":"info","params":{}}
    {"id":2,"method":"shutdown","params":{}}

Instalação: ver README na raiz.
