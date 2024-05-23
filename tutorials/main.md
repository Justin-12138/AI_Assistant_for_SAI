# Main

```latex
用法：./main [参数选择]
dw
参数选择：
  -h --help 展示帮助信息
  --version 展示版本和构建信息
  -i --interactivate 以交互模式运行
  --interactive-specials 在交互模式下允许用户文本中存在特殊标记

  --interactive-first   以交互模式运行并立即等待输入
  
  -cnv, --conversation  以对话模式运行（不打印特殊标记和后缀/前缀）(does not print special tokens and suffix/prefix)
  
  -ins, --instruct      在指令模式下运行（与 Alpaca 模型一起使用）(use with Alpaca models)
  
  -cml, --chatml        在 chatml 模式下运行(use with ChatML-compatible models)
  
  --multiline-input     允许您写入或粘贴多行，而无需以“\”结尾
  
  -r PROMPT, --reverse-prompt PROMPT
                        halt generation at PROMPT, return control in interactive mode
                        (can be specified more than once for multiple prompts).
                        
  --color 对输出进行着色以区分各代的提示和用户输入
  
  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)
  
  -t N, --threads N 生成期间所用的线程数（默认：8）
  
  -tb N, --threads-batch N 批处理和提示处理期间使用的线程数(默认: 同--threads)
                        
  -td N, --threads-draft N 生成期间使用的线程数（默认：同--threads）
  
  -tbd N, --threads-batch-draft N 批处理和提示处理期间使用的线程数（默认值：同--threads-draft）

  -p PROMPT, --prompt PROMPT 开始生成的提示词（默认：空）
                        
                        
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
  
  -ub N, --ubatch-size N physical maximum batch size (default: 512)
  
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






```

