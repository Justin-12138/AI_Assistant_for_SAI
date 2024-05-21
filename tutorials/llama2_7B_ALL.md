# llama2-7B

#### **Windows11**

+ python==3.10.8
+ torch==2.1.0
+ cmake==3.28.1
+ make==4.4.1
+ gcc version 8.1.0 (x86_64-posix-seh-rev0, Built by MinGW-W64 project)

```latex
# 创建虚拟环境
conda create -n llma2 python==3.10.8
# 启动虚拟环境
conda activate llma2
# 下载相关包
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# anaconda prompt常用密令
清屏：cls
列出当前文件夹内容：dir
复制什么的就自己使用help查看详情吧！
```

```latex
# 添加环境变量
make -v
cmake --version
gcc -v
```

模型转化，量化

```latex
# 用这位巨佬的项目
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
# 编译
cmake . -G "MinGW Makefiles"
cmake --build . --config Release
# 此时你可以看到bin目录下出现了很多exe文件，但是我们最主要使用的就main.exe，quantize.exe
```

下载模型如下,第一次下载需要申请，当然你也可以找一些非官方的的库下载，我在llama的hugging face仓库中下载对应的模型，我所使用的是 [llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b/tree/main)

![image-20240518205153181](/home/lz/Desktop/GUET_SAI_ZZ/assets/image-20240518205153181.png)

```latex
下载最主要的3个文件，在llama2.cpp中的创建一个 testmodels/llama2-7b 的文件夹并将下述3个文件放入llama2-7b中
结构如下
llama.cpp
 -...
 ...
 -testmodels
 	-llama2-7b
 		-consolidated.00.pth
		-params.json
		-tokenizer.model
```

![image-20240518205236708](/home/lz/Desktop/GUET_SAI_ZZ/assets/image-20240518205236708.png)

```latex
软件依赖安装完成！
llama项目编译成功！
模型下载完成并放到testmodels文件夹！
# 转化
cd llama.cpp
conda activate llama2
# 注意在你的虚拟环境下执行！！！
python convert.py testmodels/llama2-7b/ --outfile testmodels/llama2-7b/llama2-7b.bin
# 量化
bin\quantize.exe testmodels/llama2-7b/llama2-7b.bin testmodels/llama2-7b/llama2-7b_res.bin q4_0
# 和Bob对话
bin\main.exe -m testmodels/llama2-7b/llama2-7b_res.bin -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt

```

#### **Linux(Ubuntu22.04)**

+  make -v==GNU Make 4.3
+ gcc -v==gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04) 
+ cmake --version==cmake version 3.22.1

```latex
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
# 转化
python convert.py testmodels/llama2-7b/ --outfile testmodels/llama2-7b/llama2-7b.bin
# 量化
./quantize testmodels/llama2-7b/llama2-7b.bin testmodels/llama2-7b/llama2-7b_res.bin q4_0
# chat with bob
./main -m testmodels/llama2-7b/llama2-7b_res.bin -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt

  -m FNAME, --model FNAME
                        model path (default: models/$filename with filename from --hf-file or --model-url if set, otherwise models/7B/ggml-model-f16.gguf)


  -f FNAME, --file FNAME
                        prompt file to start generation.

  --repeat-penalty N    penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)
  
  -i, --interactive     run in interactive mode
  
  -r PROMPT, --reverse-prompt PROMPT
                        halt generation at PROMPT, return control in interactive mode
                        (can be specified more than once for multiple prompts).
  --color               colorise output to distinguish prompt and user input from generations

  -n N, --n-predict N   number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)

```

#### quantize量化参数详解

```latex
(base) lz@lz:~/Desktop/llama.cpp$ ./quantize --help
usage: ./quantize [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights] [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--override-kv] model-f32.gguf [model-quant.gguf] type [nthreads]

  --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit
  --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing
  --pure: Disable k-quant mixtures and quantize all tensors to the same type
  --imatrix file_name: use data in file_name as importance matrix for quant optimizations
  --include-weights tensor_name: use importance matrix for this/these tensor(s)
  --exclude-weights tensor_name: use importance matrix for this/these tensor(s)
  --output-tensor-type ggml_type: use this ggml_type for the output.weight tensor
  --token-embedding-type ggml_type: use this ggml_type for the token embeddings tensor
  --keep-split: will generate quatized model in the same shards as input  --override-kv KEY=TYPE:VALUE
      Advanced option to override model metadata by key in the quantized model. May be specified multiple times.
Note: --include-weights and --exclude-weights cannot be used together

Allowed quantization types:
   2  or  Q4_0    :  3.56G, +0.2166 ppl @ LLaMA-v1-7B
   3  or  Q4_1    :  3.90G, +0.1585 ppl @ LLaMA-v1-7B
   8  or  Q5_0    :  4.33G, +0.0683 ppl @ LLaMA-v1-7B
   9  or  Q5_1    :  4.70G, +0.0349 ppl @ LLaMA-v1-7B
  19  or  IQ2_XXS :  2.06 bpw quantization
  20  or  IQ2_XS  :  2.31 bpw quantization
  28  or  IQ2_S   :  2.5  bpw quantization
  29  or  IQ2_M   :  2.7  bpw quantization
  24  or  IQ1_S   :  1.56 bpw quantization
  31  or  IQ1_M   :  1.75 bpw quantization
  10  or  Q2_K    :  2.63G, +0.6717 ppl @ LLaMA-v1-7B
  21  or  Q2_K_S  :  2.16G, +9.0634 ppl @ LLaMA-v1-7B
  23  or  IQ3_XXS :  3.06 bpw quantization
  26  or  IQ3_S   :  3.44 bpw quantization
  27  or  IQ3_M   :  3.66 bpw quantization mix
  12  or  Q3_K    : alias for Q3_K_M
  22  or  IQ3_XS  :  3.3 bpw quantization
  11  or  Q3_K_S  :  2.75G, +0.5551 ppl @ LLaMA-v1-7B
  12  or  Q3_K_M  :  3.07G, +0.2496 ppl @ LLaMA-v1-7B
  13  or  Q3_K_L  :  3.35G, +0.1764 ppl @ LLaMA-v1-7B
  25  or  IQ4_NL  :  4.50 bpw non-linear quantization
  30  or  IQ4_XS  :  4.25 bpw non-linear quantization
  15  or  Q4_K    : alias for Q4_K_M
  14  or  Q4_K_S  :  3.59G, +0.0992 ppl @ LLaMA-v1-7B
  15  or  Q4_K_M  :  3.80G, +0.0532 ppl @ LLaMA-v1-7B
  17  or  Q5_K    : alias for Q5_K_M
  16  or  Q5_K_S  :  4.33G, +0.0400 ppl @ LLaMA-v1-7B
  17  or  Q5_K_M  :  4.45G, +0.0122 ppl @ LLaMA-v1-7B
  18  or  Q6_K    :  5.15G, +0.0008 ppl @ LLaMA-v1-7B
   7  or  Q8_0    :  6.70G, +0.0004 ppl @ LLaMA-v1-7B
   1  or  F16     : 14.00G, -0.0020 ppl @ Mistral-7B
  32  or  BF16    : 14.00G, -0.0050 ppl @ Mistral-7B
   0  or  F32     : 26.00G              @ 7B
          COPY    : only copy tensors, no quantizing
```

#### main参数详细解释

```latex
(base) lz@lz:~/Desktop/llama.cpp$ ./main --help

usage: ./main [options]

options:
  -h, --help            show this help message and exit
  --version             show version and build info
  -i, --interactive     run in interactive mode
  --interactive-specials allow special tokens in user text, in interactive mode
  --interactive-first   run in interactive mode and wait for input right away
  -cnv, --conversation  run in conversation mode (does not print special tokens and suffix/prefix)
  -ins, --instruct      run in instruction mode (use with Alpaca models)
  -cml, --chatml        run in chatml mode (use with ChatML-compatible models)
  --multiline-input     allows you to write or paste multiple lines without ending each in '\'
  -r PROMPT, --reverse-prompt PROMPT
                        halt generation at PROMPT, return control in interactive mode
                        (can be specified more than once for multiple prompts).
  --color               colorise output to distinguish prompt and user input from generations
  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)
  -t N, --threads N     number of threads to use during generation (default: 8)
  -tb N, --threads-batch N
                        number of threads to use during batch and prompt processing (default: same as --threads)
  -td N, --threads-draft N                        number of threads to use during generation (default: same as --threads)
  -tbd N, --threads-batch-draft N
                        number of threads to use during batch and prompt processing (default: same as --threads-draft)
  -p PROMPT, --prompt PROMPT
                        prompt to start generation with (default: empty)
  -e, --escape          process prompt escapes sequences (\n, \r, \t, \', \", \\)
  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)
  --prompt-cache-all    if specified, saves user input and generations to cache as well.
                        not supported with --interactive or other interactive options
  --prompt-cache-ro     if specified, uses the prompt cache but does not update it.
  --random-prompt       start with a randomized prompt.
  --in-prefix-bos       prefix BOS to user inputs, preceding the `--in-prefix` string
  --in-prefix STRING    string to prefix user inputs with (default: empty)
  --in-suffix STRING    string to suffix after user inputs with (default: empty)
  -f FNAME, --file FNAME
                        prompt file to start generation.
  -bf FNAME, --binary-file FNAME
                        binary file containing multiple choice tasks.
  -n N, --n-predict N   number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
  -c N, --ctx-size N    size of the prompt context (default: 512, 0 = loaded from model)
  -b N, --batch-size N  logical maximum batch size (default: 2048)
  -ub N, --ubatch-size N
                        physical maximum batch size (default: 512)
  --samplers            samplers that will be used for generation in the order, separated by ';'
                        (default: top_k;tfs_z;typical_p;top_p;min_p;temperature)
  --sampling-seq        simplified sequence for samplers that will be used (default: kfypmt)
  --top-k N             top-k sampling (default: 40, 0 = disabled)
  --top-p N             top-p sampling (default: 0.9, 1.0 = disabled)
  --min-p N             min-p sampling (default: 0.1, 0.0 = disabled)
  --tfs N               tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
  --typical N           locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
  --repeat-last-n N     last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)
  --repeat-penalty N    penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)
  --presence-penalty N  repeat alpha presence penalty (default: 0.0, 0.0 = disabled)
  --frequency-penalty N repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)
  --dynatemp-range N    dynamic temperature range (default: 0.0, 0.0 = disabled)
  --dynatemp-exp N      dynamic temperature exponent (default: 1.0)
  --mirostat N          use Mirostat sampling.
                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.
                        (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
  --mirostat-lr N       Mirostat learning rate, parameter eta (default: 0.1)
  --mirostat-ent N      Mirostat target entropy, parameter tau (default: 5.0)
  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS
                        modifies the likelihood of token appearing in the completion,
                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',
                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'
  --grammar GRAMMAR     BNF-like grammar to constrain generations (see samples in grammars/ dir)
  --grammar-file FNAME  file to read grammar from
  -j SCHEMA, --json-schema SCHEMA
                        JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object.
                        For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead
  --cfg-negative-prompt PROMPT
                        negative prompt to use for guidance. (default: empty)
  --cfg-negative-prompt-file FNAME
                        negative prompt file to use for guidance. (default: empty)
  --cfg-scale N         strength of guidance (default: 1.000000, 1.0 = disable)
  --rope-scaling {none,linear,yarn}
                        RoPE frequency scaling method, defaults to linear unless specified by the model
  --rope-scale N        RoPE context scaling factor, expands context by a factor of N
  --rope-freq-base N    RoPE base frequency, used by NTK-aware scaling (default: loaded from model)
  --rope-freq-scale N   RoPE frequency scaling factor, expands context by a factor of 1/N
  --yarn-orig-ctx N     YaRN: original context size of model (default: 0 = model training context size)
  --yarn-ext-factor N   YaRN: extrapolation mix factor (default: 1.0, 0.0 = full interpolation)
  --yarn-attn-factor N  YaRN: scale sqrt(t) or attention magnitude (default: 1.0)
  --yarn-beta-slow N    YaRN: high correction dim or alpha (default: 1.0)
  --yarn-beta-fast N    YaRN: low correction dim or beta (default: 32.0)
  --pooling {none,mean,cls}
                        pooling type for embeddings, use model default if unspecified
  -dt N, --defrag-thold N
                        KV cache defragmentation threshold (default: -1.0, < 0 - disabled)
  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)
  --penalize-nl         penalize newline tokens
  --temp N              temperature (default: 0.8)
  --all-logits          return logits for all tokens in the batch (default: disabled)
  --hellaswag           compute HellaSwag score over random tasks from datafile supplied with -f
  --hellaswag-tasks N   number of tasks to use when computing the HellaSwag score (default: 400)
  --winogrande          compute Winogrande score over random tasks from datafile supplied with -f
  --winogrande-tasks N  number of tasks to use when computing the Winogrande score (default: 0)
  --multiple-choice     compute multiple choice score over random tasks from datafile supplied with -f
  --multiple-choice-tasks N number of tasks to use when computing the multiple choice score (default: 0)
  --kl-divergence       computes KL-divergence to logits provided via --kl-divergence-base
  --keep N              number of tokens to keep from the initial prompt (default: 0, -1 = all)
  --draft N             number of tokens to draft for speculative decoding (default: 5)
  --chunks N            max number of chunks to process (default: -1, -1 = all)
  -np N, --parallel N   number of parallel sequences to decode (default: 1)
  -ns N, --sequences N  number of sequences to decode (default: 1)
  -ps N, --p-split N    speculative decoding split probability (default: 0.1)
  -cb, --cont-batching  enable continuous batching (a.k.a dynamic batching) (default: disabled)
  -fa, --flash-attn     enable Flash Attention (default: disabled)
  --mmproj MMPROJ_FILE  path to a multimodal projector file for LLaVA. see examples/llava/README.md
  --image IMAGE_FILE    path to an image file. use with multimodal models. Specify multiple times for batching
  --mlock               force system to keep model in RAM rather than swapping or compressing
  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)
  --numa TYPE           attempt optimizations that help on some NUMA systems
                          - distribute: spread execution evenly over all nodes
                          - isolate: only spawn threads on CPUs on the node that execution started on
                          - numactl: use the CPU map provided by numactl
                        if run without this previously, it is recommended to drop the system page cache before using this
                        see https://github.com/ggerganov/llama.cpp/issues/1437
  --rpc SERVERS         comma separated list of RPC servers
  --verbose-prompt      print a verbose prompt before generation (default: false)
  --no-display-prompt   don't print prompt at generation (default: false)
  -gan N, --grp-attn-n N
                        group-attention factor (default: 1)
  -gaw N, --grp-attn-w N
                        group-attention width (default: 512.0)
  -dkvc, --dump-kv-cache
                        verbose print of the KV cache
  -nkvo, --no-kv-offload
                        disable KV offload
  -ctk TYPE, --cache-type-k TYPE
                        KV cache data type for K (default: f16)
  -ctv TYPE, --cache-type-v TYPE
                        KV cache data type for V (default: f16)
  --simple-io           use basic IO for better compatibility in subprocesses and limited consoles
  --lora FNAME          apply LoRA adapter (implies --no-mmap)
  --lora-scaled FNAME S apply LoRA adapter with user defined scaling S (implies --no-mmap)
  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter
  --control-vector FNAME
                        add a control vector
  --control-vector-scaled FNAME S
                        add a control vector with user defined scaling S
  --control-vector-layer-range START END
                        layer range to apply the control vector(s) to, start and end inclusive
  -m FNAME, --model FNAME
                        model path (default: models/$filename with filename from --hf-file or --model-url if set, otherwise models/7B/ggml-model-f16.gguf)
  -md FNAME, --model-draft FNAME
                        draft model for speculative decoding (default: unused)
  -mu MODEL_URL, --model-url MODEL_URL
                        model download url (default: unused)
  -hfr REPO, --hf-repo REPO
                        Hugging Face model repository (default: unused)
  -hff FILE, --hf-file FILE
                        Hugging Face model file (default: unused)
  -ld LOGDIR, --logdir LOGDIR
                        path under which to save YAML logs (no logging if unset)
  -lcs FNAME, --lookup-cache-static FNAME
                        path to static lookup cache to use for lookup decoding (not updated by generation)
  -lcd FNAME, --lookup-cache-dynamic FNAME
                        path to dynamic lookup cache to use for lookup decoding (updated by generation)
  --override-kv KEY=TYPE:VALUE
                        advanced option to override model metadata by key. may be specified multiple times.
                        types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false
  -ptc N, --print-token-count N
                        print token count every N tokens (default: -1)
  --check-tensors       check model tensor data for invalid values

log options:
  --log-test            Run simple logging test
  --log-disable         Disable trace logs
  --log-enable          Enable trace logs
  --log-file            Specify a log filename (without extension)
  --log-new             Create a separate new log file on start. Each log file will have unique name: "<name>.<ID>.log"
  --log-append          Don't truncate the old log file.
```





#### server参数详解

```latex
(base) lz@lz:~/Desktop/llama.cpp$ ./server --help
usage: ./server [options]

options:
  -h, --help                show this help message and exit
  -v, --verbose             verbose output (default: disabled)
  -t N, --threads N         number of threads to use during computation (default: 8)
  -tb N, --threads-batch N  number of threads to use during batch and prompt processing (default: same as --threads)
  --threads-http N          number of threads in the http server pool to process requests (default: max(hardware concurrency - 1, --parallel N + 2))
  -c N, --ctx-size N        size of the prompt context (default: 512)
  --rope-scaling {none,linear,yarn}
                            RoPE frequency scaling method, defaults to linear unless specified by the model
  --rope-freq-base N        RoPE base frequency (default: loaded from model)
  --rope-freq-scale N       RoPE frequency scaling factor, expands context by a factor of 1/N
  --yarn-ext-factor N       YaRN: extrapolation mix factor (default: 1.0, 0.0 = full interpolation)
  --yarn-attn-factor N      YaRN: scale sqrt(t) or attention magnitude (default: 1.0)
  --yarn-beta-slow N        YaRN: high correction dim or alpha (default: 1.0)
  --yarn-beta-fast N        YaRN: low correction dim or beta (default: 32.0)
  --pooling {none,mean,cls} pooling type for embeddings, use model default if unspecified
  -dt N, --defrag-thold N
                            KV cache defragmentation threshold (default: -1.0, < 0 - disabled)
  -b N, --batch-size N      logical maximum batch size (default: 2048)
  -ub N, --ubatch-size N    physical maximum batch size (default: 512)
  --mlock                   force system to keep model in RAM rather than swapping or compressing
  --no-mmap                 do not memory-map model (slower load but may reduce pageouts if not using mlock)
  --numa TYPE               attempt optimizations that help on some NUMA systems
                              - distribute: spread execution evenly over all nodes
                              - isolate: only spawn threads on CPUs on the node that execution started on
                              - numactl: use the CPU map provided my numactl
  -m FNAME, --model FNAME
                            model path (default: models/$filename with filename from --hf-file or --model-url if set, otherwise models/7B/ggml-model-f16.gguf)
  -mu MODEL_URL, --model-url MODEL_URL
                            model download url (default: unused)
  -hfr REPO, --hf-repo REPO
                            Hugging Face model repository (default: unused)
  -hff FILE, --hf-file FILE
                            Hugging Face model file (default: unused)
  -a ALIAS, --alias ALIAS
                            set an alias for the model, will be added as `model` field in completion response
  --lora FNAME              apply LoRA adapter (implies --no-mmap)
  --lora-base FNAME         optional model to use as a base for the layers modified by the LoRA adapter
  --host                    ip address to listen (default  (default: 127.0.0.1)
  --port PORT               port to listen (default  (default: 8080)
  --rpc SERVERS             comma separated list of RPC servers
  --path PUBLIC_PATH        path from which to serve static files (default: disabled)
  --api-key API_KEY         optional api key to enhance server security. If set, requests must include this key for access.
  --api-key-file FNAME      path to file containing api keys delimited by new lines. If set, requests must include one of the keys for access.
  -to N, --timeout N        server read/write timeout in seconds (default: 600)
  --embeddings              enable embedding vector output (default: disabled)
  -np N, --parallel N       number of slots for process requests (default: 1)
  -cb, --cont-batching      enable continuous batching (a.k.a dynamic batching) (default: enabled)
  -fa, --flash-attn         enable Flash Attention (default: disabled)
  -spf FNAME, --system-prompt-file FNAME
                            set a file to load a system prompt (initial prompt of all slots), this is useful for chat applications.
  -ctk TYPE, --cache-type-k TYPE
                            KV cache data type for K (default: f16)
  -ctv TYPE, --cache-type-v TYPE
                            KV cache data type for V (default: f16)
  --log-format              log output format: json or text (default: json)
  --log-disable             disables logging to a file.
  --slots-endpoint-disable  disables slots monitoring endpoint.
  --metrics                 enable prometheus compatible metrics endpoint (default: disabled).
  --slot-save-path PATH     path to save slot kv cache (default: disabled)

  -n, --n-predict           maximum tokens to predict (default: -1)
  --override-kv KEY=TYPE:VALUE
                            advanced option to override model metadata by key. may be specified multiple times.
                            types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false
  -gan N, --grp-attn-n N    set the group attention factor to extend context size through self-extend(default: 1=disabled), used together with group attention width `--grp-attn-w`
  -gaw N, --grp-attn-w N    set the group attention width to extend context size through self-extend(default: 512), used together with group attention factor `--grp-attn-n`
  --chat-template JINJA_TEMPLATE
                            set custom jinja chat template (default: template taken from model's metadata)
                            only commonly used templates are accepted:
                            https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template

```



#### 微调参数详解

```latex
(base) lz@lz:~/Desktop/llama.cpp$ ./finetune -help
error: unknown argument: -help
usage: ./finetune [options]

options:
  -h, --help                 show this help message and exit
  --model-base FNAME         model path from which to load base model (default '')
  --lora-out FNAME           path to save llama lora (default 'ggml-lora-ITERATION-f32.gguf')
  --only-write-lora          only save llama lora, don't do any training.  use this if you only want to convert a checkpoint to a lora adapter.
  
  --norm-rms-eps F           RMS-Norm epsilon value (default 0.000010)
  --rope-freq-base F         Frequency base for ROPE (default 10000.000000)
  --rope-freq-scale F        Frequency scale for ROPE (default 1.000000)
  --lora-alpha N             LORA alpha : resulting LORA scaling is alpha/r. (default 4)
  --lora-r N                 LORA r: default rank. Also specifies resulting scaling together with lora-alpha. (default 4)
  --rank-att-norm N          LORA rank for attention norm tensor, overrides default rank. Norm tensors should generally have rank 1.
  --rank-ffn-norm N          LORA rank for feed-forward norm tensor, overrides default rank. Norm tensors should generally have rank 1.
  --rank-out-norm N          LORA rank for output norm tensor, overrides default rank. Norm tensors should generally have rank 1.
  --rank-tok-embd N          LORA rank for token embeddings tensor, overrides default rank.
  --rank-out N               LORA rank for output tensor, overrides default rank.
  --rank-wq N                LORA rank for wq tensor, overrides default rank.
  --rank-wk N                LORA rank for wk tensor, overrides default rank.
  --rank-wv N                LORA rank for wv tensor, overrides default rank.
  --rank-wo N                LORA rank for wo tensor, overrides default rank.
  --rank-ffn_gate N          LORA rank for ffn_gate tensor, overrides default rank.
  --rank-ffn_down N          LORA rank for ffn_down tensor, overrides default rank.
  --rank-ffn_up N            LORA rank for ffn_up tensor, overrides default rank.
  --train-data FNAME         path from which to load training data (default 'shakespeare.txt')
  --checkpoint-in FNAME      path from which to load training checkpoint (default 'checkpoint.gguf')
  --checkpoint-out FNAME     path to save training checkpoint (default 'checkpoint-ITERATION.gguf')
  --pattern-fn-it STR        pattern in output filenames to be replaced by iteration number (default 'ITERATION')
  --fn-latest STR            string to use instead of iteration number for saving latest output (default 'LATEST')
  --save-every N             save checkpoint and lora every N iterations. Disabled when N <= 0. (default '10')
  -s SEED, --seed SEED       RNG seed (default: -1, use random seed for -1)
  -c N, --ctx N              Context size used during training (default 128)
  -t N, --threads N          Number of threads (default 6)
  -b N, --batch N            Parallel batch size (default 8)
  --grad-acc N               Number of gradient accumulation steps (simulates larger batch size of batch*gradacc) (default 1)
  --sample-start STR         Sets the starting point for samples after the specified pattern. If empty use every token position as sample start. (default '')
  --include-sample-start     Include the sample start in the samples. (default off)
  --escape                   process sample start escapes sequences (\n, \r, \t, \', \", \\)
  --overlapping-samples      Samples may overlap, will include sample-start of second and following samples. When off, samples will end at begin of next sample. (default off)
  --fill-with-next-samples   Samples shorter than context length will be followed by the next (shuffled) samples. (default off)
  --separate-with-eos        When fill-with-next-samples, insert end-of-sequence token between samples.
  --separate-with-bos        When fill-with-next-samples, insert begin-of-sequence token between samples. (default)
  --no-separate-with-eos     When fill-with-next-samples, don't insert end-of-sequence token between samples. (default)
  --no-separate-with-bos     When fill-with-next-samples, don't insert begin-of-sequence token between samples.
  --sample-random-offsets    Use samples beginning at random offsets. Together with fill-with-next-samples this may help for training endless text generation.
  --force-reshuffle          Force a reshuffling of data at program start, otherwise the shuffling of loaded checkpoint is resumed.
  --no-flash                 Don't use flash attention 
  --use-flash                Use flash attention (default)
  --no-checkpointing         Don't use gradient checkpointing
  --use-checkpointing        Use gradient checkpointing (default)
  --warmup N                 Only for Adam optimizer. Number of warmup steps (default 100)
  --cos-decay-steps N        Only for Adam optimizer. Number of cosine decay steps (default 1000)
  --cos-decay-restart N      Only for Adam optimizer. Increase of cosine decay steps after restart (default 1.100000)
  --cos-decay-min N          Only for Adam optimizer. Cosine decay minimum (default 0.100000)
  --enable-restart N         Only for Adam optimizer. Enable restarts of cos-decay 
  --disable-restart N        Only for Adam optimizer. Disable restarts of cos-decay (default)
  --opt-past N               Number of optimization iterations to track for delta convergence test. Disabled when zero. (default 0)
  --opt-delta N              Maximum delta for delta convergence test. Disabled when <= zero. (default 0.000010)
  --opt-max-no-improvement N Maximum number of optimization iterations with no improvement. Disabled when <= zero. (default 0)
  --epochs N                 Maximum number epochs to process. (default -1)
  --adam-iter N              Maximum number of Adam optimization iterations for each batch (default 256)
  --adam-alpha N             Adam learning rate alpha (default 0.001000)
  --adam-min-alpha N         Adam minimum learning rate alpha - including warmup phase (default 0.000000)
  --adam-decay N             AdamW weight decay. Values greater zero enable AdamW instead of regular Adam. (default 0.100000)
  --adam-decay-min-ndim N    Minimum number of tensor dimensions to apply AdamW weight decay. Weight decay is not applied to tensors with less n_dims. (default 2)
  --adam-beta1 N             AdamW beta1 in interval [0,1). How much to smooth the first moment of gradients. (default 0.900000)
  --adam-beta2 N             AdamW beta2 in interval [0,1). How much to smooth the second moment of gradients. (default 0.999000)
  --adam-gclip N             AdamW gradient clipping. Disabled when zero. (default 1.000000)
  --adam-epsf N              AdamW epsilon for convergence test. Disabled when <= zero. (default 0.000000)
  -ngl N, --n-gpu-layers N   Number of model layers to offload to GPU (default 0)

```



#### 并行参数详细解释

```latex
(base) lz@lz:~/Desktop/llama.cpp$ ./parallel -help
error: unknown argument: -help

usage: ./parallel [options]

options:
  -h, --help            show this help message and exit
  --version             show version and build info
  -i, --interactive     run in interactive mode
  --interactive-specials allow special tokens in user text, in interactive mode
  --interactive-first   run in interactive mode and wait for input right away
  -cnv, --conversation  run in conversation mode (does not print special tokens and suffix/prefix)
  -ins, --instruct      run in instruction mode (use with Alpaca models)
  -cml, --chatml        run in chatml mode (use with ChatML-compatible models)
  --multiline-input     allows you to write or paste multiple lines without ending each in '\'
  -r PROMPT, --reverse-prompt PROMPT
                        halt generation at PROMPT, return control in interactive mode
                        (can be specified more than once for multiple prompts).
  --color               colorise output to distinguish prompt and user input from generations
  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)
  -t N, --threads N     number of threads to use during generation (default: 8)
  -tb N, --threads-batch N
                        number of threads to use during batch and prompt processing (default: same as --threads)
  -td N, --threads-draft N                        number of threads to use during generation (default: same as --threads)
  -tbd N, --threads-batch-draft N
                        number of threads to use during batch and prompt processing (default: same as --threads-draft)
  -p PROMPT, --prompt PROMPT
                        prompt to start generation with (default: empty)
  -e, --escape          process prompt escapes sequences (\n, \r, \t, \', \", \\)
  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)
  --prompt-cache-all    if specified, saves user input and generations to cache as well.
                        not supported with --interactive or other interactive options
  --prompt-cache-ro     if specified, uses the prompt cache but does not update it.
  --random-prompt       start with a randomized prompt.
  --in-prefix-bos       prefix BOS to user inputs, preceding the `--in-prefix` string
  --in-prefix STRING    string to prefix user inputs with (default: empty)
  --in-suffix STRING    string to suffix after user inputs with (default: empty)
  -f FNAME, --file FNAME
                        prompt file to start generation.
  -bf FNAME, --binary-file FNAME
                        binary file containing multiple choice tasks.
  -n N, --n-predict N   number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
  -c N, --ctx-size N    size of the prompt context (default: 512, 0 = loaded from model)
  -b N, --batch-size N  logical maximum batch size (default: 2048)
  -ub N, --ubatch-size N
                        physical maximum batch size (default: 512)
  --samplers            samplers that will be used for generation in the order, separated by ';'
                        (default: top_k;tfs_z;typical_p;top_p;min_p;temperature)
  --sampling-seq        simplified sequence for samplers that will be used (default: kfypmt)
  --top-k N             top-k sampling (default: 40, 0 = disabled)
  --top-p N             top-p sampling (default: 0.9, 1.0 = disabled)
  --min-p N             min-p sampling (default: 0.1, 0.0 = disabled)
  --tfs N               tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
  --typical N           locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
  --repeat-last-n N     last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)
  --repeat-penalty N    penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)
  --presence-penalty N  repeat alpha presence penalty (default: 0.0, 0.0 = disabled)
  --frequency-penalty N repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)
  --dynatemp-range N    dynamic temperature range (default: 0.0, 0.0 = disabled)
  --dynatemp-exp N      dynamic temperature exponent (default: 1.0)
  --mirostat N          use Mirostat sampling.
                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.
                        (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
  --mirostat-lr N       Mirostat learning rate, parameter eta (default: 0.1)
  --mirostat-ent N      Mirostat target entropy, parameter tau (default: 5.0)
  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS
                        modifies the likelihood of token appearing in the completion,
                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',
                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'
  --grammar GRAMMAR     BNF-like grammar to constrain generations (see samples in grammars/ dir)
  --grammar-file FNAME  file to read grammar from
  -j SCHEMA, --json-schema SCHEMA
                        JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object.
                        For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead
  --cfg-negative-prompt PROMPT
                        negative prompt to use for guidance. (default: empty)
  --cfg-negative-prompt-file FNAME
                        negative prompt file to use for guidance. (default: empty)
  --cfg-scale N         strength of guidance (default: 1.000000, 1.0 = disable)
  --rope-scaling {none,linear,yarn}
                        RoPE frequency scaling method, defaults to linear unless specified by the model
  --rope-scale N        RoPE context scaling factor, expands context by a factor of N
  --rope-freq-base N    RoPE base frequency, used by NTK-aware scaling (default: loaded from model)
  --rope-freq-scale N   RoPE frequency scaling factor, expands context by a factor of 1/N
  --yarn-orig-ctx N     YaRN: original context size of model (default: 0 = model training context size)
  --yarn-ext-factor N   YaRN: extrapolation mix factor (default: 1.0, 0.0 = full interpolation)
  --yarn-attn-factor N  YaRN: scale sqrt(t) or attention magnitude (default: 1.0)
  --yarn-beta-slow N    YaRN: high correction dim or alpha (default: 1.0)
  --yarn-beta-fast N    YaRN: low correction dim or beta (default: 32.0)
  --pooling {none,mean,cls}
                        pooling type for embeddings, use model default if unspecified
  -dt N, --defrag-thold N
                        KV cache defragmentation threshold (default: -1.0, < 0 - disabled)
  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)
  --penalize-nl         penalize newline tokens
  --temp N              temperature (default: 0.8)
  --all-logits          return logits for all tokens in the batch (default: disabled)
  --hellaswag           compute HellaSwag score over random tasks from datafile supplied with -f
  --hellaswag-tasks N   number of tasks to use when computing the HellaSwag score (default: 400)
  --winogrande          compute Winogrande score over random tasks from datafile supplied with -f
  --winogrande-tasks N  number of tasks to use when computing the Winogrande score (default: 0)
  --multiple-choice     compute multiple choice score over random tasks from datafile supplied with -f
  --multiple-choice-tasks N number of tasks to use when computing the multiple choice score (default: 0)
  --kl-divergence       computes KL-divergence to logits provided via --kl-divergence-base
  --keep N              number of tokens to keep from the initial prompt (default: 0, -1 = all)
  --draft N             number of tokens to draft for speculative decoding (default: 5)
  --chunks N            max number of chunks to process (default: -1, -1 = all)
  -np N, --parallel N   number of parallel sequences to decode (default: 1)
  -ns N, --sequences N  number of sequences to decode (default: 1)
  -ps N, --p-split N    speculative decoding split probability (default: 0.1)
  -cb, --cont-batching  enable continuous batching (a.k.a dynamic batching) (default: disabled)
  -fa, --flash-attn     enable Flash Attention (default: disabled)
  --mmproj MMPROJ_FILE  path to a multimodal projector file for LLaVA. see examples/llava/README.md
  --image IMAGE_FILE    path to an image file. use with multimodal models. Specify multiple times for batching
  --mlock               force system to keep model in RAM rather than swapping or compressing
  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)
  --numa TYPE           attempt optimizations that help on some NUMA systems
                          - distribute: spread execution evenly over all nodes
                          - isolate: only spawn threads on CPUs on the node that execution started on
                          - numactl: use the CPU map provided by numactl
                        if run without this previously, it is recommended to drop the system page cache before using this
                        see https://github.com/ggerganov/llama.cpp/issues/1437
  --rpc SERVERS         comma separated list of RPC servers
  --verbose-prompt      print a verbose prompt before generation (default: false)
  --no-display-prompt   don't print prompt at generation (default: false)
  -gan N, --grp-attn-n N
                        group-attention factor (default: 1)
  -gaw N, --grp-attn-w N
                        group-attention width (default: 512.0)
  -dkvc, --dump-kv-cache
                        verbose print of the KV cache
  -nkvo, --no-kv-offload
                        disable KV offload
  -ctk TYPE, --cache-type-k TYPE
                        KV cache data type for K (default: f16)
  -ctv TYPE, --cache-type-v TYPE
                        KV cache data type for V (default: f16)
  --simple-io           use basic IO for better compatibility in subprocesses and limited consoles
  --lora FNAME          apply LoRA adapter (implies --no-mmap)
  --lora-scaled FNAME S apply LoRA adapter with user defined scaling S (implies --no-mmap)
  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter
  --control-vector FNAME
                        add a control vector
  --control-vector-scaled FNAME S
                        add a control vector with user defined scaling S
  --control-vector-layer-range START END
                        layer range to apply the control vector(s) to, start and end inclusive
  -m FNAME, --model FNAME
                        model path (default: models/$filename with filename from --hf-file or --model-url if set, otherwise models/7B/ggml-model-f16.gguf)
  -md FNAME, --model-draft FNAME
                        draft model for speculative decoding (default: unused)
  -mu MODEL_URL, --model-url MODEL_URL
                        model download url (default: unused)
  -hfr REPO, --hf-repo REPO
                        Hugging Face model repository (default: unused)
  -hff FILE, --hf-file FILE
                        Hugging Face model file (default: unused)
  -ld LOGDIR, --logdir LOGDIR
                        path under which to save YAML logs (no logging if unset)
  -lcs FNAME, --lookup-cache-static FNAME
                        path to static lookup cache to use for lookup decoding (not updated by generation)
  -lcd FNAME, --lookup-cache-dynamic FNAME
                        path to dynamic lookup cache to use for lookup decoding (updated by generation)
  --override-kv KEY=TYPE:VALUE
                        advanced option to override model metadata by key. may be specified multiple times.
                        types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false
  -ptc N, --print-token-count N
                        print token count every N tokens (default: -1)
  --check-tensors       check model tensor data for invalid values

log options:
  --log-test            Run simple logging test
  --log-disable         Disable trace logs
  --log-enable          Enable trace logs
  --log-file            Specify a log filename (without extension)
  --log-new             Create a separate new log file on start. Each log file will have unique name: "<name>.<ID>.log"
  --log-append          Don't truncate the old log file.

```

#### retrieval参数详解

```latex
(base) lz@lz:~/Desktop/llama.cpp$ ./retrieval -help
error: unknown argument: -help

usage: ./retrieval [options]

options:
  -h, --help            show this help message and exit
  --version             show version and build info
  -i, --interactive     run in interactive mode
  --interactive-specials allow special tokens in user text, in interactive mode
  --interactive-first   run in interactive mode and wait for input right away
  -cnv, --conversation  run in conversation mode (does not print special tokens and suffix/prefix)
  -ins, --instruct      run in instruction mode (use with Alpaca models)
  -cml, --chatml        run in chatml mode (use with ChatML-compatible models)
  --multiline-input     allows you to write or paste multiple lines without ending each in '\'
  -r PROMPT, --reverse-prompt PROMPT
                        halt generation at PROMPT, return control in interactive mode
                        (can be specified more than once for multiple prompts).
  --color               colorise output to distinguish prompt and user input from generations
  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)
  -t N, --threads N     number of threads to use during generation (default: 8)
  -tb N, --threads-batch N
                        number of threads to use during batch and prompt processing (default: same as --threads)
  -td N, --threads-draft N                        number of threads to use during generation (default: same as --threads)
  -tbd N, --threads-batch-draft N
                        number of threads to use during batch and prompt processing (default: same as --threads-draft)
  -p PROMPT, --prompt PROMPT
                        prompt to start generation with (default: empty)
  -e, --escape          process prompt escapes sequences (\n, \r, \t, \', \", \\)
  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)
  --prompt-cache-all    if specified, saves user input and generations to cache as well.
                        not supported with --interactive or other interactive options
  --prompt-cache-ro     if specified, uses the prompt cache but does not update it.
  --random-prompt       start with a randomized prompt.
  --in-prefix-bos       prefix BOS to user inputs, preceding the `--in-prefix` string
  --in-prefix STRING    string to prefix user inputs with (default: empty)
  --in-suffix STRING    string to suffix after user inputs with (default: empty)
  -f FNAME, --file FNAME
                        prompt file to start generation.
  -bf FNAME, --binary-file FNAME
                        binary file containing multiple choice tasks.
  -n N, --n-predict N   number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
  -c N, --ctx-size N    size of the prompt context (default: 512, 0 = loaded from model)
  -b N, --batch-size N  logical maximum batch size (default: 2048)
  -ub N, --ubatch-size N
                        physical maximum batch size (default: 512)
  --samplers            samplers that will be used for generation in the order, separated by ';'
                        (default: top_k;tfs_z;typical_p;top_p;min_p;temperature)
  --sampling-seq        simplified sequence for samplers that will be used (default: kfypmt)
  --top-k N             top-k sampling (default: 40, 0 = disabled)
  --top-p N             top-p sampling (default: 0.9, 1.0 = disabled)
  --min-p N             min-p sampling (default: 0.1, 0.0 = disabled)
  --tfs N               tail free sampling, parameter z (default: 1.0, 1.0 = disabled)
  --typical N           locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
  --repeat-last-n N     last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)
  --repeat-penalty N    penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)
  --presence-penalty N  repeat alpha presence penalty (default: 0.0, 0.0 = disabled)
  --frequency-penalty N repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)
  --dynatemp-range N    dynamic temperature range (default: 0.0, 0.0 = disabled)
  --dynatemp-exp N      dynamic temperature exponent (default: 1.0)
  --mirostat N          use Mirostat sampling.
                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.
                        (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
  --mirostat-lr N       Mirostat learning rate, parameter eta (default: 0.1)
  --mirostat-ent N      Mirostat target entropy, parameter tau (default: 5.0)
  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS
                        modifies the likelihood of token appearing in the completion,
                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',
                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'
  --grammar GRAMMAR     BNF-like grammar to constrain generations (see samples in grammars/ dir)
  --grammar-file FNAME  file to read grammar from
  -j SCHEMA, --json-schema SCHEMA
                        JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object.
                        For schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead
  --cfg-negative-prompt PROMPT
                        negative prompt to use for guidance. (default: empty)
  --cfg-negative-prompt-file FNAME
                        negative prompt file to use for guidance. (default: empty)
  --cfg-scale N         strength of guidance (default: 1.000000, 1.0 = disable)
  --rope-scaling {none,linear,yarn}
                        RoPE frequency scaling method, defaults to linear unless specified by the model
  --rope-scale N        RoPE context scaling factor, expands context by a factor of N
  --rope-freq-base N    RoPE base frequency, used by NTK-aware scaling (default: loaded from model)
  --rope-freq-scale N   RoPE frequency scaling factor, expands context by a factor of 1/N
  --yarn-orig-ctx N     YaRN: original context size of model (default: 0 = model training context size)
  --yarn-ext-factor N   YaRN: extrapolation mix factor (default: 1.0, 0.0 = full interpolation)
  --yarn-attn-factor N  YaRN: scale sqrt(t) or attention magnitude (default: 1.0)
  --yarn-beta-slow N    YaRN: high correction dim or alpha (default: 1.0)
  --yarn-beta-fast N    YaRN: low correction dim or beta (default: 32.0)
  --pooling {none,mean,cls}
                        pooling type for embeddings, use model default if unspecified
  -dt N, --defrag-thold N
                        KV cache defragmentation threshold (default: -1.0, < 0 - disabled)
  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)
  --penalize-nl         penalize newline tokens
  --temp N              temperature (default: 0.8)
  --all-logits          return logits for all tokens in the batch (default: disabled)
  --hellaswag           compute HellaSwag score over random tasks from datafile supplied with -f
  --hellaswag-tasks N   number of tasks to use when computing the HellaSwag score (default: 400)
  --winogrande          compute Winogrande score over random tasks from datafile supplied with -f
  --winogrande-tasks N  number of tasks to use when computing the Winogrande score (default: 0)
  --multiple-choice     compute multiple choice score over random tasks from datafile supplied with -f
  --multiple-choice-tasks N number of tasks to use when computing the multiple choice score (default: 0)
  --kl-divergence       computes KL-divergence to logits provided via --kl-divergence-base
  --keep N              number of tokens to keep from the initial prompt (default: 0, -1 = all)
  --draft N             number of tokens to draft for speculative decoding (default: 5)
  --chunks N            max number of chunks to process (default: -1, -1 = all)
  -np N, --parallel N   number of parallel sequences to decode (default: 1)
  -ns N, --sequences N  number of sequences to decode (default: 1)
  -ps N, --p-split N    speculative decoding split probability (default: 0.1)
  -cb, --cont-batching  enable continuous batching (a.k.a dynamic batching) (default: disabled)
  -fa, --flash-attn     enable Flash Attention (default: disabled)
  --mmproj MMPROJ_FILE  path to a multimodal projector file for LLaVA. see examples/llava/README.md
  --image IMAGE_FILE    path to an image file. use with multimodal models. Specify multiple times for batching
  --mlock               force system to keep model in RAM rather than swapping or compressing
  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)
  --numa TYPE           attempt optimizations that help on some NUMA systems
                          - distribute: spread execution evenly over all nodes
                          - isolate: only spawn threads on CPUs on the node that execution started on
                          - numactl: use the CPU map provided by numactl
                        if run without this previously, it is recommended to drop the system page cache before using this
                        see https://github.com/ggerganov/llama.cpp/issues/1437
  --rpc SERVERS         comma separated list of RPC servers
  --verbose-prompt      print a verbose prompt before generation (default: false)
  --no-display-prompt   don't print prompt at generation (default: false)
  -gan N, --grp-attn-n N
                        group-attention factor (default: 1)
  -gaw N, --grp-attn-w N
                        group-attention width (default: 512.0)
  -dkvc, --dump-kv-cache
                        verbose print of the KV cache
  -nkvo, --no-kv-offload
                        disable KV offload
  -ctk TYPE, --cache-type-k TYPE
                        KV cache data type for K (default: f16)
  -ctv TYPE, --cache-type-v TYPE
                        KV cache data type for V (default: f16)
  --simple-io           use basic IO for better compatibility in subprocesses and limited consoles
  --lora FNAME          apply LoRA adapter (implies --no-mmap)
  --lora-scaled FNAME S apply LoRA adapter with user defined scaling S (implies --no-mmap)
  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter
  --control-vector FNAME
                        add a control vector
  --control-vector-scaled FNAME S
                        add a control vector with user defined scaling S
  --control-vector-layer-range START END
                        layer range to apply the control vector(s) to, start and end inclusive
  -m FNAME, --model FNAME
                        model path (default: models/$filename with filename from --hf-file or --model-url if set, otherwise models/7B/ggml-model-f16.gguf)
  -md FNAME, --model-draft FNAME
                        draft model for speculative decoding (default: unused)
  -mu MODEL_URL, --model-url MODEL_URL
                        model download url (default: unused)
  -hfr REPO, --hf-repo REPO
                        Hugging Face model repository (default: unused)
  -hff FILE, --hf-file FILE
                        Hugging Face model file (default: unused)
  -ld LOGDIR, --logdir LOGDIR
                        path under which to save YAML logs (no logging if unset)
  -lcs FNAME, --lookup-cache-static FNAME
                        path to static lookup cache to use for lookup decoding (not updated by generation)
  -lcd FNAME, --lookup-cache-dynamic FNAME
                        path to dynamic lookup cache to use for lookup decoding (updated by generation)
  --override-kv KEY=TYPE:VALUE
                        advanced option to override model metadata by key. may be specified multiple times.
                        types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false
  -ptc N, --print-token-count N
                        print token count every N tokens (default: -1)
  --check-tensors       check model tensor data for invalid values

log options:
  --log-test            Run simple logging test
  --log-disable         Disable trace logs
  --log-enable          Enable trace logs
  --log-file            Specify a log filename (without extension)
  --log-new             Create a separate new log file on start. Each log file will have unique name: "<name>.<ID>.log"
  --log-append          Don't truncate the old log file.

retrieval options:
  --context-file FNAME  file containing context to embed.
                        specify multiple files by providing --context-file option multiple times.
  --chunk-size N        minimum length of embedded text chunk (default:64)
  --chunk-separator STRING
                        string to separate chunks (default: "\n")

```

