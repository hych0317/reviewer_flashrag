python ./run_reviewer.py \
    --method_name 'reviewer_reasoning' \
    --dataset_name 'triviaqa' \
    --split 'train' \
    --gpu_id 1 \
    --max_iter 2 \
    --max_docs 8 \
    --generator_model 'qwen3-4B-Instruct' \
    --sample_size 1000
    # Full data: --sample_size 0
    # NQ: --dataset_name nq --split test
    # PopQA: --dataset_name popqa --split test
    # HotpotQA: --dataset_name hotpotqa --split dev/train
    # 2WikiMultiHopQA: --dataset_name 2wikimultihopqa --split dev/train
    # MuSiQue: --dataset_name musique --split dev/train
    # TriviaQA: --dataset_name triviaqa --split test/dev/train
    # StrategyQA: --dataset_name strategyqa --split train
    # AmbigQA: --dataset_name ambig_qa --split dev/train
    # ASQA: --dataset_name asqa --split dev/train
    # FEVER: --dataset_name fever --split dev/train
