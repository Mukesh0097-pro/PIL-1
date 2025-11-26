# PIL-VAE Hybrid Engine

## The Technical Hook
"We use **Cython** to transpile our proprietary math kernels into **C++**, and then compile that to **WebAssembly**. This allows our `core/` engine to run entirely in the client's browser with near-native performance, eliminating server GPU costs."

## The Concept: "Ahead-of-Time" (AOT) Compilation

Standard Python (what you use in development) is **Interpreted**. It reads your code line-by-line while it runs. This is flexible but slow.
* **The "Compile to C++" part:** You take your Python math logic (the `core/` folder) and translate it into C++ code, which runs 10-100x faster because it talks directly to the hardware.
* **The "Compile to WebAssembly" part:** You take that C++ code and turn it into a `.wasm` binary file. This binary can run inside any web browser at near-native speed.

## How We Do It (The Pipeline)

### Path A: The "Logic" Pipeline (For Custom Math/PIL-VAE)
The `PILVAEDecoder` is just linear algebra (Matrix Multiplication, SVD). We turn this Python logic into a high-performance binary.

1.  **Tool:** **Cython** or **Nuitka**
    * *Input:* `engine.py` (Python)
    * *Action:* Translates Python variables into C++ types (e.g., `numpy.array` becomes `std::vector`).
    * *Output:* `engine.cpp` (High-performance C++ source code).
2.  **Tool:** **Emscripten**
    * *Input:* `engine.cpp`
    * *Action:* Compiles the C++ code into `engine.wasm`.
    * *Result:* A binary file that runs exact math logic in the browser, but 50x faster than Python.

### Path B: The "Model" Pipeline (For the Transformer)
For the neural network part (the "Reading Brain"), we export the *graph*.

1.  **Tool:** **ONNX (Open Neural Network Exchange)**
    * *Action:* "Trace" the data flowing through the PyTorch model and freeze it into a static file (`model.onnx`).
2.  **Runtime:** **ONNX Runtime Web**
    * *Action:* The browser uses a pre-built WebAssembly engine to execute this graph. It uses the user's laptop CPU or GPU (via WebGL) to run the math.

## Business Value
* **Zero Server Cost:** Computation happens on the *user's* device (Client-Side).
* **Total Privacy:** Data never leaves the device.
* **Offline Capability:** Works without internet once loaded.
