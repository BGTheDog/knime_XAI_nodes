# Grok nodes for KNIME

Python KNIME extension from **Whozinzahs Holdings, LLC** ([Whozinzahs.com](https://Whozinzahs.com)) that talks to the [xAI API](https://docs.x.ai) (`https://api.x.ai/v1`). Nodes use **Grok-only port types**, so they connect to each other and not to KNIME’s OpenAI / Claude / LLM Prompter ports.

## Nodes

| Node | What it does |
|---|---|
| **Grok Authenticator** | Takes an xAI API key (paste it, or use Credentials Configuration) and outputs a Grok Authentication port. |
| **Grok Chat Model Selector** | Picks a Grok model (default `grok-4.6`) and sampling settings. |
| **Grok LLM Prompter** | Sends each row of a prompt column to Grok and appends a response column. |
| **Grok Agent** | One-shot agent: prompt table in, conversation table out, optional dynamic data tables. |

Typical flow:

```
Credentials Configuration  →  Grok Authenticator  →  Grok Chat Model Selector
                                                         ├→ Grok LLM Prompter
                                                         └→ Grok Agent
```

## Agent data ports

On **Grok Agent**, use the node’s **⋯ → Add output port** (and add input port) controls — the same UI as KNIME’s Agent Prompter and Python Script.

- **Data inputs**: tables the agent can inspect and query.
- **Data outputs**: tables the agent can fill. Add as many as you expect, then ask for that data in the prompt (for example “put a two-column summary on data output 0”).

Ports are not created automatically from the prompt. Add them first, then execute.

## Develop in KNIME Analytics Platform

Requires KNIME 5.9+ with **KNIME Python Extension Development**.

```bash
pixi install
pixi run register-debug-in-knime
```

Or point `knime.ini` at this folder:

```
-Dknime.python.extension.config=/absolute/path/to/knime_dev/config.yml
```

and set absolute paths inside `config.yml` for `src` and `conda_env_path`.

Fully **quit and reopen KNIME Analytics Platform 5.9** (not just close the workspace). Search the Node Repository for `Grok`. The four nodes sit in a top-level **Grok** category.

Get an API key at [console.x.ai](https://console.x.ai). Paste it in **Grok Authenticator**, or put it in the **password** field of a Credentials Configuration node (username is ignored) and select those credentials under Advanced settings.

## Tests

```bash
pixi run test
```

## Bundle a local update site

```bash
pixi run build
```

Then in KNIME: **File → Install KNIME Extensions… → Available Software Sites → Add…** and choose `./local-update-site`.

## Share with other KNIME trainers

Two packages, depending on the audience.

**Trainers who only need the nodes in class**

1. `pixi run --environment build build`
2. Zip `local-update-site/` and send it.
3. Recipients: KNIME **5.9+**, **File → Install KNIME Extensions… → Available Software Sites → Add…** → the unzipped folder → install **Grok**.
4. Each person uses their own xAI API key. Do not ship a key in the Authenticator or in a demo workflow.

**Trainer-developers who will change the nodes**

Send the git repo (or a zip of the tree **without** `.pixi/`, `config.yml`, and `local-update-site/`). They run:

```bash
pixi install
pixi run test
pixi run register-debug-in-knime
```

Then fully quit and reopen KNIME 5.9. Demo workflow: Grok Authenticator → Grok Chat Model Selector → Table Creator (prompt column) → Grok LLM Prompter.

KNIME 5.4 / 5.7 will not pick this up unless you also register those installs; develop and demo against **5.9**.

## License

Copyright 2026 Whozinzahs Holdings, LLC. Licensed under the Apache License, Version 2.0. See `LICENSE.TXT`.
