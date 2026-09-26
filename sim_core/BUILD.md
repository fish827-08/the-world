# sim_core 构建说明（本机 Windows）

> 建立：2026-09-26（`[所有者·天平]` 本地验证 `sim_core/src/lib.rs` 改动时踩坑记录）
> 用途：**本机有 Rust 工具链**（此前误判为"本机无 cargo"，实为不在 bash `PATH` 里）。

## 1. 工具链在哪

```
C:\Users\圣羽\.cargo\bin\cargo.exe      # 不在 bash PATH ⇒ `which cargo` 会失败！
C:\Users\圣羽\.rustup\toolchains\stable-x86_64-pc-windows-msvc
maturin 1.15.0（已装在 .venv）
```

⇒ **`which cargo` 返回空 ≠ 没装 Rust**。要先用绝对路径或补 PATH。

## 2. 常规构建（推荐）

```
cd sim_core
..\.venv\Scripts\python.exe -m maturin develop --release
```

⚠️ maturin 需要 `rustc` 在 PATH ⇒ Git Bash 下先 `PATH="/c/Users/圣羽/.cargo/bin:$PATH"`。

## 3. 🔴 只做「编译/类型自检」的绕过配方（**最有用**）

`pyo3-build-config` 的 build script 会 **spawn 一个 Python 子进程**去探测解释器；
在受限执行环境（沙箱）里会失败：

```
error: failed to run the Python interpreter at ...: 所有的管道范例都在使用中。(os error 231)
```

此时即使设 `PYO3_PYTHON=<venv python>` 也无效（它仍要 spawn）。
**绕过办法：完全不让它见解释器 —— `PYO3_NO_PYTHON=1` + 一个 abi3 特性**：

```
cd sim_core
PYO3_NO_PYTHON=1 cargo check --release \
    --features pyo3/extension-module,pyo3/abi3-py314
```

（`abi3` 特性是必须的，否则报
"An abi3-py3* feature must be specified when compiling without a Python interpreter"。）

- ✅ 实测：**13.2 s 通过**，产出与真正 build 相同的**语法/类型**诊断。
- ⚠️ **局限**：只做 `check`（不链接、不产出 `.pyd`）⇒ **不能替代**真正的 `maturin develop`；
  用途是"快速证明我改的 Rust 能编译"，避免把编译错误推给别人。

## 4. 谁该关心

* 改 `sim_core/src/*.rs` 之后，**提交前**至少跑 §3 自检。
* 只有需要真正出 `.pyd`（跑 Rust 路径测试）时才用 §2。
