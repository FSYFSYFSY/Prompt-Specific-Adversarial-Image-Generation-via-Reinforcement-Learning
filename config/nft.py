import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


def _get_config(
    base_model="sd3",
    n_gpus=1,
    gradient_step_per_epoch=1,
    dataset="safebench",
    reward_fn={},
    name="",
    num_image_per_prompt=8,
    num_groups=24,
    bsz=9,
):
    config = base.get_config()
    assert base_model in ["sd3"]
    assert dataset in ["safebench", "pickscore", "ocr", "geneval"]

    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    if base_model == "sd3":
        config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 4.5
        config.resolution = 512
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7

    # 组大小 k 与 batch 几何（三者耦和，改一个就要重算）：
    #   - bsz 必须 >= num_image_per_prompt，否则组内归一化拿不到完整的组；
    #   - bsz % num_image_per_prompt == 0；
    #   - num_groups * num_image_per_prompt % (n_gpus * bsz) == 0。
    # 注意下面的 while 是从 bsz 起【递减】搜索的，所以 bsz 必须 >= 你想要的 batch。
    config.sample.num_image_per_prompt = num_image_per_prompt

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    # special design, the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = 14 if dataset == "geneval" else 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval" if dataset == "geneval" else "general_ocr"

    config.run_name = f"nft_{base_model}_{name}"
    config.save_dir = f"logs/nft/{base_model}/{name}"
    config.reward_fn = reward_fn

    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"

    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config


def sd3_ocr():
    reward_fn = {
        "ocr": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=8, gradient_step_per_epoch=2, dataset="ocr", reward_fn=reward_fn, name="ocr"
    )
    config.beta = 0.1
    config.decay_type = 2
    return config


def sd3_geneval():
    reward_fn = {
        "geneval": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="geneval",
    )
    return config


def sd3_pickscore():
    reward_fn = {
        "pickscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore",
    )
    return config


def sd3_hpsv2():
    reward_fn = {
        "hpsv2": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=8, gradient_step_per_epoch=1, dataset="pickscore", reward_fn=reward_fn, name="hpsv2"
    )
    return config

def sd3_jailguard(num_image_per_prompt=8, bsz=9, name="jailguard"):
    reward_fn = {
        "jailguard": 1.0,  # 权重为1.0
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=1,
        gradient_step_per_epoch=1,
        dataset="safebench",  # 根据你的prompt数据集选择
        reward_fn=reward_fn,
        name=name,
        num_image_per_prompt=num_image_per_prompt,
        num_groups=24,
        bsz=bsz,
    )
    # 启用 checkpoint 保存：debug=False 后每 save_freq 个 epoch 保存一次。
    # 保存到 logs/nft/sd3/jailguard/checkpoints/checkpoint-{seed}-{global_step}/
    # 内含 lora/ + optimizer.pt + scaler.pt + ema.pt
    config.debug = False
    config.save_freq = 10          # 每 10 个 epoch 保存一次
    # 不跑 eval：只保存 checkpoint，跳过测试集采样 + judge 打分。
    # 如需周期性评估，改为 True 并调小 eval_freq。
    config.do_eval = False
    # 避免 debug=False 后在 epoch 0（0 % eval_freq == 0）触发一次无意义 eval；
    # 如需周期性评估，把该值改小即可
    config.eval_freq = 1000000
    # 断点续训：指向某个 checkpoint 目录（例如 checkpoint-42-1000），
    # 会自动恢复 LoRA / 优化器 / scaler / EMA，并从目录名恢复种子与 global_step
    # config.resume_from = "logs/nft/sd3/jailguard/checkpoints/checkpoint-42-1000"
    # 根据你的需要调整其他超参数
    return config


def sd3_jailguard_internvl3():
    """sd3_jailguard 的蓝队换成 InternVL3-8B 版本，其余超参与旧 run 完全一致。

    为什么单独开一个配置：checkpoint 目录名是 checkpoint-{seed}-{global_step}，
    只要 seed 相同（都是 42），两个 run 就会写进同名目录、把之前用 Qwen3-VL
    当蓝队跑出来的结果覆盖掉。这里把 save_dir 挪到独立目录，旧结果原样保留。

    蓝队模型本身由环境变量控制（在 nft.sh 里 export）：
        BLUE_VLM_BASE_URL=http://127.0.0.1:17141/v1
        BLUE_VLM_MODEL=InternVL3-8B
    """
    config = sd3_jailguard()
    config.run_name = "nft_sd3_jailguard_internvl3"
    config.save_dir = "logs/nft/sd3/jailguard-internvl3"
    return config


def sd3_jailguard_llava():
    """sd3_jailguard 的蓝队换成 llava-onevision-qwen2-7b-ov，并把出图步数提到 28。

    为什么单独开一个配置：
      - 蓝队由环境变量控制（nft.sh 里 export），但 save_dir 必须区分开，
        否则会写进 checkpoint-42-* 同名目录、把别的 run 覆盖掉。
      - 出图步数：训练默认 10 步、评测流水线用 40 步
        （run_decoupled.sh --num_inference_steps 40），两者出图分布不同、分数不可比。
        这里提到 28 步以缩小差距（评测口径仍是 40，不在这里改）。

    蓝队模型本身由环境变量控制（在 nft.sh 里 export）：
        BLUE_VLM_BASE_URL=http://127.0.0.1:17141/v1
        BLUE_VLM_MODEL=llava-onevision-qwen2-7b-ov
    启动服务：bash /autodl-fs/data/sglang-models.sh llava jail
    """
    # 16 张/组（原 8）：组内样本翻倍 -> reward 非零样本的绝对数翻倍，
    # 实测有方差的组（真正产生梯度的组）从 21% 提到理论 ~57%，不再被少数几组垄断信号。
    # 约束：bsz 必须 >= num_image_per_prompt 且能被其整除，否则 _get_config 的
    # batch 搜索（从 bsz 起递减）找不到解并 assert False。这里 24*16=384，取 bsz=16。
    config = sd3_jailguard(num_image_per_prompt=16, bsz=16, name="jailguard-llava")
    # save_dir 由 name 推出 = logs/nft/sd3/jailguard-llava
    config.run_name = "nft_sd3_jailguard_llava"
    # 训练侧 micro-batch 减半回 8：bsz=16 直接跑训练前向会 OOM
    #（实测峰值 97200 MiB / 只剩 52 MiB，挂在 train_nft_sd3.py:1180 transformer_ddp）。
    # 代码支持把收集到的样本切成 num_batches 个 micro-batch 前向
    #（training_batch_size = total_batch_size_filtered // num_batches），
    # 所以「16 张一组算 advantage」和「8 张一次前向」可以并存。
    # accum 步数必须同步放大，否则每 epoch 会变成 2 次优化器更新
    #（不变式：gradient_accumulation_steps * num_train_timesteps == 本 epoch 的总反向次数）。
    config.train.batch_size = 8
    config.train.gradient_accumulation_steps = (
        config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size
    )
    # DAPO 式动态采样：丢弃零方差组（实测 llava 下 75% 的 prompt 是任何一张图
    # 都攻不破的死组，advantage 恒 0）。见 train_nft_sd3.py 里的 _select_kept。
    config.train.drop_zero_std_groups = True
    # 10 -> 28：向评测的 40 步靠拢，减少训练/评测口径差异
    config.sample.num_steps = 28
    return config


def sd3_multi_reward():
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "clipscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.sample.num_steps = 25
    config.beta = 0.1
    return config
