# Learning 01 — C++ Mental Models (from a JS/TS brain)

Goal: give you *just enough* C++ mental model to read and write StemForge confidently.
Every concept is mapped to something you already know from TypeScript/Rust.

---

## 1. There is no garbage collector. You own memory.

In JS, you make objects and forget about them. In C++, every object has an **owner**
responsible for its lifetime. This is the single biggest mental shift.

C++ gives you three ways to hold a thing:

```cpp
AudioBuffer buf;                          // (a) value — lives on the stack, dies at }
std::unique_ptr<Session> session;         // (b) owning pointer — heap, freed automatically
AudioBuffer& ref = buf;                   // (c) reference — a borrow, owns nothing
AudioBuffer* ptr = &buf;                  // (d) raw pointer — a borrow, can be null
```

- **(a) value / stack** — like a `const x = {}` that vanishes at the end of its block.
  Fast, automatic, no heap. Prefer this.
- **(b) `unique_ptr`** — exactly Rust's `Box<T>`. One owner. When the owner dies, the
  thing is freed. Move it with `std::move`, never copy it.
- **(c) reference `&`** — a non-null borrow. Like `&T` in Rust. Use for "I need to read/
  modify this but I don't own it" — e.g. function params for big buffers.
- **(d) raw pointer `*`** — a *borrow that might be null*. We use these only to talk to
  C APIs (ONNX's float arrays). Never own with a raw pointer.

> **Rule for this project:** Own with `unique_ptr`. Borrow with `&`. Touch raw `*` only
> at C-API boundaries.

---

## 2. RAII — the destructor is your `finally`

**RAII** = "Resource Acquisition Is Initialization." Translation: *cleanup is tied to
scope*. When an object goes out of scope, its **destructor** runs automatically. This is
how C++ does `finally`/`defer` without a keyword.

```cpp
{
    Ort::Value tensor = makeTensor(...);   // resource acquired
    session.Run(tensor);
}   // ← tensor's destructor runs HERE, memory freed. No leak. No manual free().
```

This is why the coding standard says "scope your `Ort::Value`." You don't free things;
you let scope free them. A `unique_ptr` member freed in a class destructor is the same
idea one level up.

---

## 3. Header (`.h`) vs source (`.cpp`) — declaration vs definition

JS has one file per module. C++ splits each unit in two:

- **`.h` (header)** — the *interface*. What exists: class shape, method signatures,
  member variables. Like a `.d.ts` file. Other files `#include` this to know the API.
- **`.cpp` (source)** — the *implementation*. The actual method bodies.

```cpp
// ONNXInferenceEngine.h  — the "what"
class ONNXInferenceEngine {
public:
    juce::Result loadModel(const juce::File& path);   // declared, no body
private:
    std::unique_ptr<Ort::Session> m_session;
};

// ONNXInferenceEngine.cpp — the "how"
#include "ONNXInferenceEngine.h"
juce::Result ONNXInferenceEngine::loadModel(const juce::File& path) {
    // ... real code ...
}
```

`#pragma once` at the top of every header stops it being pasted in twice (header guard).

---

## 4. `const` is your `readonly` — use it aggressively

`const` means "I promise not to modify this." The compiler enforces it. It's free
documentation + safety.

```cpp
void writeStems(const juce::AudioBuffer<float>& buffer) const;
//              ^^^^^ won't modify the buffer              ^^^^^ won't modify the object
```

A `const&` parameter = "pass this big object by borrow, read-only, no copy." That's the
default way to pass buffers and files around. Copying a 4-minute audio buffer by accident
is a real performance bug `const&` prevents.

---

## 5. Value semantics & the cost of `=`

In JS, `a = b` for objects copies a *reference*. In C++, `a = b` copies the *whole thing*
by default — every sample in the buffer. Three operations to know:

```cpp
auto b = a;              // COPY — duplicates all data (expensive for buffers)
auto b = std::move(a);   // MOVE — steals a's guts, a is now empty (cheap, O(1))
auto& b = a;             // REFERENCE — no copy, b is just another name for a
```

`std::move` is how you transfer a `unique_ptr` or a big buffer without copying — it's
Rust's move, made explicit. After a move, the source is empty; don't use it.

---

## 6. `std::atomic` — lock-free cross-thread variables

A plain `float m_progress` written by the worker thread and read by the UI thread is a
**data race** (undefined behavior). Wrap it:

```cpp
std::atomic<float> m_progress { 0.0f };
m_progress.store(0.5f);        // worker thread writes
float p = m_progress.load();   // UI thread reads — safe, no lock
```

This is the *only* sanctioned way to share a simple value across threads here. For
anything bigger than a scalar, use a lock or a message queue. (See `03-how-its-built.md`
for the threading model.)

---

## 7. Templates `<T>` — generics, resolved at compile time

`juce::AudioBuffer<float>` is a generic, like `Array<number>`. The `<float>` says "buffer
of 32-bit floats." `std::vector<float>` is "growable array of floats" — your `Float32Array`
equivalent, but it owns and frees its storage automatically. You'll mostly consume
templates, rarely write them.

---

## Cheat sheet

| You want… | TS/Rust analog | C++ |
|---|---|---|
| Owned heap object | `Box<T>` | `std::unique_ptr<T>` |
| Shared owned object | `Rc<T>`/`Arc<T>` | `std::shared_ptr<T>` |
| Growable array | `Array` / `Vec<T>` | `std::vector<T>` |
| Read-only borrow | `&T` | `const T&` |
| Cleanup on scope exit | `finally` / `Drop` | destructor (RAII) |
| Move ownership | `let b = a;` (Rust) | `std::move(a)` |
| Cross-thread scalar | `Arc<AtomicF32>` | `std::atomic<float>` |
| `readonly` | `readonly` / `const` | `const` |

Next: `02-data-structures.md` — the actual data shapes audio flows through.
