# Third-party notices

PdfPecker.Ocr is licensed under the MIT License (see `LICENSE`). The package bundles neural network model data that
is derived from third-party works under other licences. Those works, their licences and the changes made to them are
listed here. This file ships with the package.

## PaddleOCR models, as packaged by RapidOCR 3.10.0 (Apache License 2.0)

`src/PdfPecker.Ocr/Models/det.bin.br`, `rec.bin.br` and `cls.bin.br` are converted from these ONNX files in the
[rapidocr 3.10.0](https://pypi.org/project/rapidocr/3.10.0/) wheel (`rapidocr-3.10.0-py3-none-any.whl`, SHA-256
`2fc34e26cd0f48514804a94f8832283039c7334076bff60768b1fed9ad068050`):

| PdfPecker.Ocr file | File in the wheel | Upstream model | SHA-256 of the ONNX file |
| --- | --- | --- | --- |
| `det.bin.br` | `rapidocr/models/PP-OCRv6_det_small.onnx` | PP-OCRv6_small_det | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| `rec.bin.br` | `rapidocr/models/PP-OCRv6_rec_small.onnx` | PP-OCRv6_small_rec | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |
| `cls.bin.br` | `rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx` | ch_ppocr_mobile_v2.0_cls | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |

The upstream model weights are copyright Baidu and/or the applicable PaddleOCR rights holders; the ONNX conversions
were made by the RapidOCR Authors. Both are licensed under the Apache License, Version 2.0, reproduced below.

**These files were changed.** `tools/ModelBuilder` converted them to PdfPecker.Ocr's own model format: the graphs were
simplified (constant shape arithmetic folded, batch normalisation folded into the preceding convolutions, activations,
layer normalisation, squeeze-and-excitation and attention blocks fused into single operators), and the weights are
stored as single-precision values rounded to 24 bits. Each converted file carries a notice saying so. The weights
remain copyright of their rights holders and are licensed under the Apache License 2.0, not under the MIT License.

The model notice shipped in the wheel (`rapidocr-3.10.0.dist-info/licenses/MODEL_LICENSES.md`) follows verbatim, its
headings nested under this section.

### RapidOCR Model License and Attribution Notice

This document records the license, copyright attribution, provenance, and
redistribution requirements for OCR model artifacts distributed by RapidOCR.
It does not replace or modify the Apache License, Version 2.0.

#### 1. Separate licensed works

RapidOCR contains two separately copyrighted categories of work:

1. **Source code and engineering components.** Copyright is held by the
   RapidOCR Authors. These components are licensed under the Apache License,
   Version 2.0, as described in the RapidOCR project `LICENSE` file.
2. **OCR model artifacts.** Copyright in the upstream model weights is held by
   Baidu and/or the applicable PaddleOCR rights holders. RapidOCR does not
   claim ownership of the upstream model weights.

Both categories are distributed under the Apache License, Version 2.0, but
their copyright ownership and provenance remain separate.

#### 2. Scope of this notice

This notice applies to the model artifacts listed in Section 6. Other model
artifacts available through RapidOCR may have different upstream versions or
terms and must be reviewed against their own source records before being
redistributed.

This notice does not automatically cover third-party models, datasets, fonts,
sample images, inference runtimes, or other assets that are not derived from
official PaddleOCR models. Those works remain subject to their own applicable
terms.

#### 3. Upstream attribution and licensing evidence

The models in Section 6 are released by, or derived from models released by,
the PaddleOCR project. PaddleOCR is released under the Apache License, Version
2.0. Official PaddlePaddle model cards identify the PP-OCRv6 model weights and
ONNX representations as `Apache-2.0`. PaddleOCR's official model list
identifies the legacy text angle classification model and its official
distribution source.

Upstream references:

- PaddleOCR project: <https://github.com/PaddlePaddle/PaddleOCR>
- PaddleOCR license: <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE>
- PaddleOCR model list: <https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/model_list.md>
- PaddleOCR ONNX conversion documentation: <https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/inference_deployment/others/obtaining_onnx_models.html>
- Official PP-OCRv6 small detection ONNX model card: <https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx>
- Official PP-OCRv6 small recognition ONNX model card: <https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx>
- Official legacy text angle classification model list: <https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_list.en.md>

The legacy `ch_ppocr_mobile_v2.0_cls` model predates the official Hugging Face
model hosting and is identified through PaddleOCR's official model list and
model download service.

#### 4. Applicable license

The model artifacts in Section 6 are made available under the Apache License,
Version 2.0 (`Apache-2.0`). The complete license text is available at
<https://www.apache.org/licenses/LICENSE-2.0>.

Subject to the conditions of Apache-2.0, the license permits use,
reproduction, modification, preparation of derivative works, public display,
sublicensing, and distribution in source or object form. These permissions are
not restricted to non-commercial use.

The license applies only to rights that the relevant licensors are authorized
to grant. No additional model-specific restriction has been identified for the
PaddleOCR-derived artifacts covered by this notice.

#### 5. Converted model artifacts

RapidOCR converts or repackages upstream PaddleOCR inference models for use
with supported inference engines. These conversions are mechanical
transformations of the model representation. Conversion does not transfer
ownership of the upstream weights to RapidOCR and does not remove the upstream
copyright or attribution.

RapidOCR distributes the converted artifacts in Section 6 under the same
Apache-2.0 terms that apply to the corresponding upstream PaddleOCR models. The
conversion and packaging scripts, configuration, and RapidOCR-authored
metadata are separately copyrighted by the RapidOCR Authors and licensed under
Apache-2.0 as part of the RapidOCR source code.

A RapidOCR-converted artifact may not be byte-for-byte identical to an ONNX or
other representation hosted by PaddlePaddle. Artifact identity must therefore
be established using its source record and cryptographic hash rather than its
filename alone.

#### 6. Bundled model artifacts

The following artifacts are covered by this notice and are distributed with
RapidOCR's default ONNX Runtime configuration:

| RapidOCR artifact | Upstream model | SHA-256 | RapidOCR source |
| --- | --- | --- | --- |
| `PP-OCRv6_det_small.onnx` | `PP-OCRv6_small_det` | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` | [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx) |
| `PP-OCRv6_rec_small.onnx` | `PP-OCRv6_small_rec` | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` | [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx) |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | `ch_ppocr_mobile_v2.0_cls` | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` | [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx) |

The model registry in `rapidocr/default_models.yaml` is the authoritative
release record for the source URL and SHA-256 value of each supported model
artifact.

#### 7. Redistribution requirements

When redistributing a covered model artifact, or a modified or converted form
of one, comply with Section 4 of Apache-2.0. In particular, you must:

1. Give recipients a copy of the Apache License, Version 2.0.
2. Cause modified files to carry prominent notices stating that you changed
   them.
3. Retain applicable copyright, patent, trademark, and attribution notices
   from the upstream work, except notices that do not pertain to the
   redistributed work.
4. If an upstream distribution contains a `NOTICE` file, include the relevant
   contents of that notice in a permitted form.

At the time of publication, RapidOCR has not identified an additional
model-specific `NOTICE` file for the artifacts listed in Section 6. If a future
upstream release supplies one, its applicable contents must be retained.

For converted files, an appropriate modification notice is:

> Converted from an official PaddleOCR model to the indicated inference format
> and packaged for use with RapidOCR. The upstream model weights are copyright
> Baidu and/or the applicable PaddleOCR rights holders and are licensed under
> the Apache License, Version 2.0.

#### 8. Trademarks and endorsement

Apache-2.0 does not grant permission to use the trade names, trademarks,
service marks, or product names of Baidu, PaddlePaddle, PaddleOCR, or RapidOCR,
except as required for reasonable and customary description of the origin of
the model artifacts. Redistribution must not imply endorsement, sponsorship,
or affiliation without separate permission.

#### 9. Disclaimer

Unless required by applicable law or agreed to in writing, the model artifacts
are provided on an **AS IS** basis, without warranties or conditions of any
kind. The warranty disclaimer and limitation of liability in Apache-2.0 apply.

Model accuracy, fitness, security, and regulatory suitability remain the
responsibility of the user and downstream distributor. RapidOCR is not a
medical device, and neither the model license nor this notice represents a
certification or guarantee for medical, safety-critical, or regulated use.

#### 10. Release maintenance

When a RapidOCR release adds or changes a model artifact, maintainers should:

1. Verify the upstream model and its applicable license.
2. Record the upstream source and immutable version or revision where
   available.
3. Record the conversion tool, tool version, model format, and conversion
   options where available.
4. Record and verify the artifact SHA-256 value.
5. Preserve any applicable upstream copyright, attribution, and `NOTICE`
   material.
6. Update this notice when the covered model families or applicable terms
   change.

### Apache License 2.0

The licence text as shipped in the wheel (`rapidocr-3.10.0.dist-info/licenses/LICENSE`):

```text
Copyright (c) 2021 RapidOCR Authors. All rights reserved.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright (c) 2021 RapidAI. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
