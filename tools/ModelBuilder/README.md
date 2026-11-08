# ModelBuilder

Writes `src/PdfPecker.Ocr/Models/det.bin.br`, `rec.bin.br` and `cls.bin.br`, the neural network models PdfPecker.Ocr
embeds, from the ONNX files in the [rapidocr 3.10.0](https://pypi.org/project/rapidocr/3.10.0/) wheel on PyPI. The
generated files are never edited by hand; change this tool and run it again.

```
dotnet run -c Release --project tools/ModelBuilder -- [--wheel path] [--out dir] [--precision fp24|fp16|fp32]
```

Without `--wheel` the wheel is downloaded once from files.pythonhosted.org into `pdfpecker-ocr-modelbuilder` in the
temporary directory. The wheel and the three ONNX files are checked against pinned SHA-256 hashes (in `Program.cs`), and
the output is byte-identical on every run: the graph passes are deterministic, the writer has no timestamps, and Brotli
(quality 11, window 24) is deterministic. A run takes about a minute, most of it Brotli. After writing each model the
tool loads it with the runtime and runs it once, so a graph the runtime cannot execute fails here rather than in a test.

The models are Apache 2.0 (copyright Baidu and/or the PaddleOCR rights holders, converted by the RapidOCR Authors);
`THIRD-PARTY-NOTICES.md` reproduces their notice, and each generated file carries a notice that it was changed.

## What the builder does

1. Parses the ONNX protobuf with its own reader (`Onnx.cs`): nodes, initialisers, `Constant` nodes, the graph input
   and output and the metadata (the recogniser's `character` list).
2. Simplifies the graph (`Passes.cs`):
   - drops `Identity` nodes and evaluates nodes whose inputs are all constants;
   - runs shape inference for three input shapes that differ in every dimension the caller chooses (batch, and width
     or height) and turns every tensor whose values come out the same for all three into a constant. That folds the
     `Shape`/`Slice`/`Concat` arithmetic of the recogniser's reshapes, whose width is dynamic. A reshape's target is
     then written from the sample shapes with 0 (copy the input's dimension) and -1 (infer), and checked against all
     three samples;
   - folds `BatchNormalization` into the preceding convolution and per-channel `Add`s into biases (computed in double
     precision);
   - turns `MatMul` by a constant matrix into `Linear`;
   - recognises activations: `Relu`, `Sigmoid`, `HardSigmoid`, `x * sigmoid(x)` (Swish), `x * hardsigmoid(x)` and
     `x * clip(x + 3, 0, 6) / 6` (HardSwish), `x * (erf(x / sqrt 2) + 1) * 0.5` (GELU), and fuses them into the
     convolution or linear layer before them;
   - fuses `ReduceMean`/`Sub`/`Pow`/`ReduceMean`/`Add`/`Sqrt`/`Div`/`Mul`/`Add` into `LayerNorm`;
   - fuses channel mean, two 1x1 convolutions and the product with the input (plus the input, in the detector's neck)
     into `SqueezeExcite`;
   - fuses the attention core (reshape to heads, transpose, slices, scaled q k^T, softmax, times v, transpose,
     reshape) into `Attention`; the qkv and output projections stay `Linear` layers;
   - turns nearest-neighbour `Resize` by integer factors into upsampling, absorbed by the `Add` or `Concat` that reads
     it, so the enlarged maps are never stored;
   - makes an `Add` of a convolution or linear result and another tensor of the same shape that layer's residual.
3. Emits the runtime graph (`Emitter.cs`): 78 nodes for the detector, 78 for the recogniser and 49 for the classifier.
   Explicit pads replace `SAME_UPPER` (a constant with stride 1). Weights keep their natural layouts (convolution
   `[Cout, Cin / group, kh, kw]`, transposed convolution `[Cin, Cout, kh, kw]`, linear `[K, N]`); the runtime packs
   them into the panel layout of the matrix multiplication kernel the processor supports when the model loads, which
   takes a few milliseconds and keeps the file independent of the processor.

## Weight precision

Weights with more than 1,024 values are stored as float32 rounded to nearest even at 15 mantissa bits ("fp24", three
bytes), smaller ones (biases, norms) as float32. Weights below 2^-64 in magnitude are written as zero: the detector has
tens of thousands left over from channels training switched off, many of them subnormal, and arithmetic on subnormals
is very slow on x86 while these values cannot change a result (the runtime flushes them on loading too). Half precision would halve the large tensors but does not meet the
1e-3 tolerance against ONNX Runtime: with every weight rounded to fp16 ONNX Runtime itself moves the detector's
probability map by up to 6.7e-3 and the recogniser's softmax by up to 9.8e-3 on real text, and rounding even single
tensors to fp16 moves the detector's output by up to 9e-3. With fp24 the differences are 1.8e-4 and 2.4e-4.
`--precision fp16` writes half precision anyway (the runtime reads all three), for size experiments.

| Model | Values | float32 | fp24 file | Brotli |
| --- | ---: | ---: | ---: | ---: |
| det | 2,453,833 | 9.8 MB | 7,390,181 B | 4,940,206 B |
| rec | 5,266,550 | 21.1 MB | 15,902,704 B | 11,748,712 B |
| cls | 127,220 | 0.5 MB | 401,338 B | 321,152 B |

## File format (version 1)

The file is Brotli-compressed; inside, all integers are little-endian. `src/PdfPecker.Ocr/Models/ModelFormat.cs` is the
single definition of the layout and of the operator codes, used by this tool, the runtime and the tests.

```
magic        8 bytes "PPOCRNN\0"
version      u16 (1)
name         i32 byte length + UTF-8          e.g. "PP-OCRv6_small_det"
notice       i32 byte length + UTF-8          origin, licence and change notice
input shape  i32 count + i32 dims             -1 where the caller chooses the size
weights      i32 count, then for each:
               u8 storage                     1 float32, 2 fp24, 3 float16
               i32 rank + i32 dims
               i32 value count
               data in byte planes: for float32 four planes (bits 31-24 of every value, then 23-16, 15-8, 7-0), for
               fp24 the first three of those, for float16 two (high byte, low byte). Planes keep the sign and exponent
               bytes together, which roughly doubles what Brotli saves.
nodes        i32 count, then for each:
               u8 operator (OpKind)
               i32 count + i32 input values   value 0 is the model input; node i writes value i + 1
               i32 output value
               i32 count + i32 integer attributes
               u8 count + f32 float attributes
               i32 count + i32 weight indices
characters   i32 count, then i32 byte length + UTF-8 of the classes joined by "\n" (recogniser only: index 0 is the
             CTC blank as an empty string, the last is the space rapidocr appends)
```

The model output is the last node's output. Attributes are positional; `OpKind` in `ModelFormat.cs` lists them for each
operator.
