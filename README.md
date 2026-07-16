# Apex Lab

A laboratory for experimentation and development.

## Getting Started

This repository is set up for collaborative development and exploration.

## Historical Data CLI

The historical data engine is available from the command line via `scripts/data.py`.

```bash
python scripts/data.py refresh-instruments

python scripts/data.py download \
    --symbol BANKNIFTY \
    --interval 30minute \
    --from 2016-01-01

python scripts/data.py update \
    --symbol BANKNIFTY \
    --interval 30minute
```

## Project Structure

- `/docs` - Documentation
- `/src` - Source code
- `/tests` - Test files

## Contributing

Contributions are welcome! Please ensure code follows the project conventions.
