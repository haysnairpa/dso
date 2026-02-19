from __future__ import annotations

from dso_symbol_api.app import app


def main() -> None:
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
