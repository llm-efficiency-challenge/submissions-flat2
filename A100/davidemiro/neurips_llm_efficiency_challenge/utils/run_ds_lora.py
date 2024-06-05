{"payload":{"allShortcutsEnabled":false,"fileTree":{"training":{"items":[{"name":"configs","path":"training/configs","contentType":"directory"},{"name":"preprocessing","path":"training/preprocessing","contentType":"directory"},{"name":"scripts","path":"training/scripts","contentType":"directory"},{"name":"utils","path":"training/utils","contentType":"directory"},{"name":"accelerate-tpu-bert-text-classification.ipynb","path":"training/accelerate-tpu-bert-text-classification.ipynb","contentType":"file"},{"name":"deepseed-falcon-180b-lora-fa.ipynb","path":"training/deepseed-falcon-180b-lora-fa.ipynb","contentType":"file"},{"name":"deepseed-flan-t5-summarization.ipynb","path":"training/deepseed-flan-t5-summarization.ipynb","contentType":"file"},{"name":"flan-t5-samsum-summarization.ipynb","path":"training/flan-t5-samsum-summarization.ipynb","contentType":"file"},{"name":"instruction-tune-llama-2-int4.ipynb","path":"training/instruction-tune-llama-2-int4.ipynb","contentType":"file"},{"name":"optimize-llama-2-gptq.ipynb","path":"training/optimize-llama-2-gptq.ipynb","contentType":"file"},{"name":"peft-flan-t5-int8-summarization.ipynb","path":"training/peft-flan-t5-int8-summarization.ipynb","contentType":"file"},{"name":"pytorch-2-0-bert-text-classification.ipynb","path":"training/pytorch-2-0-bert-text-classification.ipynb","contentType":"file"},{"name":"run_ds_lora.py","path":"training/run_ds_lora.py","contentType":"file"}],"totalCount":13},"":{"items":[{"name":"assets","path":"assets","contentType":"directory"},{"name":"container","path":"container","contentType":"directory"},{"name":"inference","path":"inference","contentType":"directory"},{"name":"training","path":"training","contentType":"directory"},{"name":".gitignore","path":".gitignore","contentType":"file"},{"name":"LICENSE","path":"LICENSE","contentType":"file"},{"name":"README.md","path":"README.md","contentType":"file"}],"totalCount":7}},"fileTreeProcessingTime":5.29765,"foldersToFetch":[],"reducedMotionEnabled":null,"repo":{"id":574983692,"defaultBranch":"main","name":"deep-learning-pytorch-huggingface","ownerLogin":"philschmid","currentUserCanPush":false,"isFork":false,"isEmpty":false,"createdAt":"2022-12-06T13:57:47.000Z","ownerAvatar":"https://avatars.githubusercontent.com/u/32632186?v=4","public":true,"private":false,"isOrgOwned":false},"symbolsExpanded":false,"treeExpanded":true,"refInfo":{"name":"main","listCacheKey":"v0:1695216811.0","canEdit":false,"refType":"branch","currentOid":"5b2a6e96af93d58d63f26ae909e73cf4c808dcb2"},"path":"training/run_ds_lora.py","currentUser":null,"blob":{"rawLines":["from dataclasses import dataclass, field","from typing import cast","","import os","import subprocess","from typing import Optional","import torch","","from transformers import HfArgumentParser, TrainingArguments, Trainer","from utils.peft_utils import SaveDeepSpeedPeftModelCallback, create_and_prepare_model","from dataset import load_from_disk","","","# Define and parse arguments.","@dataclass","class ScriptArguments:","    \"\"\"","    Additional arguments for training, which are not part of TrainingArguments.","    \"\"\"","    model_id: str = field(","      metadata={","            \"help\": \"The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc.\"","        },","    )","    dataset_path: Optional[str] = field(","        default=\"timdettmers/openassistant-guanaco\",","        metadata={\"help\": \"The preference dataset to use.\"},","    )","    lora_alpha: Optional[int] = field(default=16)","    lora_dropout: Optional[float] = field(default=0.1)","    lora_r: Optional[int] = field(default=64)","    use_flash_attn: Optional[bool] = field(","        default=False,","        metadata={\"help\": \"Enables Flash attention for training.\"},","    )","    merge_adapters: bool = field(","        metadata={\"help\": \"Wether to merge weights for LoRA.\"},","        default=False,","    )","","","def training_function(script_args:ScriptArguments, training_args:TrainingArguments):","","    # Load processed dataset from disk","    dataset = load_from_disk(script_args.dataset_path)","    ","    # Load and create peft model","    model, peft_config, tokenizer = create_and_prepare_model(script_args.model_id,training_args, script_args)","    model.config.use_cache = False","","","    # Create trainer and add callbacks","    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)","    trainer.accelerator.print(f\"{trainer.model}\")","    trainer.model.print_trainable_parameters()","    trainer.add_callback(SaveDeepSpeedPeftModelCallback(trainer, save_steps=training_args.save_steps))","    ","    # Start training","    trainer.train()","","    # Save model on main process","    trainer.accelerator.wait_for_everyone()","    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)","    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)","    if trainer.accelerator.is_main_process:","        unwrapped_model.save_pretrained(training_args.output_dir, state_dict=state_dict)","    trainer.accelerator.wait_for_everyone()","","    # TODO: add merge adapters","    # Save everything else on main process","    if trainer.args.process_index == 0:","        if script_args.merge_adapters:","            # merge adapter weights with base model and save","            # save int 4 model","            trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)","            # clear memory","            del model","            del trainer","            torch.cuda.empty_cache()","","            from peft import AutoPeftModelForCausalLM","","            # load PEFT model in fp16","            model = AutoPeftModelForCausalLM.from_pretrained(","                training_args.output_dir,","                low_cpu_mem_usage=True,","                torch_dtype=torch.float16,","            )  ","            # Merge LoRA and base model and save","            model = model.merge_and_unload()        ","            model.save_pretrained(","                training_args.output_dir, safe_serialization=True, max_shard_size=\"8GB\"","            )","        else:","            trainer.model.save_pretrained(","                training_args.output_dir, safe_serialization=True","            )","","        # save tokenizer ","        tokenizer.save_pretrained(training_args.output_dir)","","","def main():","    parser = HfArgumentParser([ScriptArguments,TrainingArguments])","    script_args, training_args = parser.parse_args_into_dataclasses()","    script_args = cast(ScriptArguments, script_args)","    training_args = cast(TrainingArguments, training_args)","    ","    training_function(script_args, training_args)","","","if __name__ == \"__main__\":","    main()"],"stylingDirectives":[[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":16,"cssClass":"pl-s1"},{"start":17,"end":23,"cssClass":"pl-k"},{"start":24,"end":33,"cssClass":"pl-s1"},{"start":35,"end":40,"cssClass":"pl-s1"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":18,"cssClass":"pl-k"},{"start":19,"end":23,"cssClass":"pl-s1"}],[],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":9,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":17,"cssClass":"pl-s1"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":18,"cssClass":"pl-k"},{"start":19,"end":27,"cssClass":"pl-v"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":12,"cssClass":"pl-s1"}],[],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":17,"cssClass":"pl-s1"},{"start":18,"end":24,"cssClass":"pl-k"},{"start":25,"end":41,"cssClass":"pl-v"},{"start":43,"end":60,"cssClass":"pl-v"},{"start":62,"end":69,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":10,"cssClass":"pl-s1"},{"start":11,"end":21,"cssClass":"pl-s1"},{"start":22,"end":28,"cssClass":"pl-k"},{"start":29,"end":59,"cssClass":"pl-v"},{"start":61,"end":85,"cssClass":"pl-s1"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":13,"cssClass":"pl-s1"},{"start":14,"end":20,"cssClass":"pl-k"},{"start":21,"end":35,"cssClass":"pl-s1"}],[],[],[{"start":0,"end":29,"cssClass":"pl-c"}],[{"start":0,"end":10,"cssClass":"pl-en"},{"start":1,"end":10,"cssClass":"pl-s1"}],[{"start":0,"end":5,"cssClass":"pl-k"},{"start":6,"end":21,"cssClass":"pl-v"}],[{"start":4,"end":7,"cssClass":"pl-s"}],[{"start":0,"end":79,"cssClass":"pl-s"}],[{"start":0,"end":7,"cssClass":"pl-s"}],[{"start":4,"end":12,"cssClass":"pl-s1"},{"start":14,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":25,"cssClass":"pl-en"}],[{"start":6,"end":14,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"}],[{"start":12,"end":18,"cssClass":"pl-s"},{"start":20,"end":112,"cssClass":"pl-s"}],[],[],[{"start":4,"end":16,"cssClass":"pl-s1"},{"start":18,"end":26,"cssClass":"pl-v"},{"start":27,"end":30,"cssClass":"pl-s1"},{"start":32,"end":33,"cssClass":"pl-c1"},{"start":34,"end":39,"cssClass":"pl-en"}],[{"start":8,"end":15,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":16,"end":51,"cssClass":"pl-s"}],[{"start":8,"end":16,"cssClass":"pl-s1"},{"start":16,"end":17,"cssClass":"pl-c1"},{"start":18,"end":24,"cssClass":"pl-s"},{"start":26,"end":58,"cssClass":"pl-s"}],[],[{"start":4,"end":14,"cssClass":"pl-s1"},{"start":16,"end":24,"cssClass":"pl-v"},{"start":25,"end":28,"cssClass":"pl-s1"},{"start":30,"end":31,"cssClass":"pl-c1"},{"start":32,"end":37,"cssClass":"pl-en"},{"start":38,"end":45,"cssClass":"pl-s1"},{"start":45,"end":46,"cssClass":"pl-c1"},{"start":46,"end":48,"cssClass":"pl-c1"}],[{"start":4,"end":16,"cssClass":"pl-s1"},{"start":18,"end":26,"cssClass":"pl-v"},{"start":27,"end":32,"cssClass":"pl-s1"},{"start":34,"end":35,"cssClass":"pl-c1"},{"start":36,"end":41,"cssClass":"pl-en"},{"start":42,"end":49,"cssClass":"pl-s1"},{"start":49,"end":50,"cssClass":"pl-c1"},{"start":50,"end":53,"cssClass":"pl-c1"}],[{"start":4,"end":10,"cssClass":"pl-s1"},{"start":12,"end":20,"cssClass":"pl-v"},{"start":21,"end":24,"cssClass":"pl-s1"},{"start":26,"end":27,"cssClass":"pl-c1"},{"start":28,"end":33,"cssClass":"pl-en"},{"start":34,"end":41,"cssClass":"pl-s1"},{"start":41,"end":42,"cssClass":"pl-c1"},{"start":42,"end":44,"cssClass":"pl-c1"}],[{"start":4,"end":18,"cssClass":"pl-s1"},{"start":20,"end":28,"cssClass":"pl-v"},{"start":29,"end":33,"cssClass":"pl-s1"},{"start":35,"end":36,"cssClass":"pl-c1"},{"start":37,"end":42,"cssClass":"pl-en"}],[{"start":8,"end":15,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":16,"end":21,"cssClass":"pl-c1"}],[{"start":8,"end":16,"cssClass":"pl-s1"},{"start":16,"end":17,"cssClass":"pl-c1"},{"start":18,"end":24,"cssClass":"pl-s"},{"start":26,"end":65,"cssClass":"pl-s"}],[],[{"start":4,"end":18,"cssClass":"pl-s1"},{"start":20,"end":24,"cssClass":"pl-s1"},{"start":25,"end":26,"cssClass":"pl-c1"},{"start":27,"end":32,"cssClass":"pl-en"}],[{"start":8,"end":16,"cssClass":"pl-s1"},{"start":16,"end":17,"cssClass":"pl-c1"},{"start":18,"end":24,"cssClass":"pl-s"},{"start":26,"end":61,"cssClass":"pl-s"}],[{"start":8,"end":15,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":16,"end":21,"cssClass":"pl-c1"}],[],[],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":21,"cssClass":"pl-en"},{"start":22,"end":33,"cssClass":"pl-s1"},{"start":34,"end":49,"cssClass":"pl-v"},{"start":51,"end":64,"cssClass":"pl-s1"},{"start":65,"end":82,"cssClass":"pl-v"}],[],[{"start":4,"end":38,"cssClass":"pl-c"}],[{"start":4,"end":11,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":14,"end":28,"cssClass":"pl-en"},{"start":29,"end":40,"cssClass":"pl-s1"},{"start":41,"end":53,"cssClass":"pl-s1"}],[],[{"start":4,"end":32,"cssClass":"pl-c"}],[{"start":4,"end":9,"cssClass":"pl-s1"},{"start":11,"end":22,"cssClass":"pl-s1"},{"start":24,"end":33,"cssClass":"pl-s1"},{"start":34,"end":35,"cssClass":"pl-c1"},{"start":36,"end":60,"cssClass":"pl-en"},{"start":61,"end":72,"cssClass":"pl-s1"},{"start":73,"end":81,"cssClass":"pl-s1"},{"start":82,"end":95,"cssClass":"pl-s1"},{"start":97,"end":108,"cssClass":"pl-s1"}],[{"start":4,"end":9,"cssClass":"pl-s1"},{"start":10,"end":16,"cssClass":"pl-s1"},{"start":17,"end":26,"cssClass":"pl-s1"},{"start":27,"end":28,"cssClass":"pl-c1"},{"start":29,"end":34,"cssClass":"pl-c1"}],[],[],[{"start":4,"end":38,"cssClass":"pl-c"}],[{"start":4,"end":11,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":14,"end":21,"cssClass":"pl-v"},{"start":22,"end":27,"cssClass":"pl-s1"},{"start":27,"end":28,"cssClass":"pl-c1"},{"start":28,"end":33,"cssClass":"pl-s1"},{"start":35,"end":39,"cssClass":"pl-s1"},{"start":39,"end":40,"cssClass":"pl-c1"},{"start":40,"end":53,"cssClass":"pl-s1"},{"start":55,"end":68,"cssClass":"pl-s1"},{"start":68,"end":69,"cssClass":"pl-c1"},{"start":69,"end":76,"cssClass":"pl-s1"}],[{"start":4,"end":11,"cssClass":"pl-s1"},{"start":12,"end":23,"cssClass":"pl-s1"},{"start":24,"end":29,"cssClass":"pl-en"},{"start":30,"end":48,"cssClass":"pl-s"},{"start":32,"end":47,"cssClass":"pl-s1"},{"start":32,"end":33,"cssClass":"pl-kos"},{"start":33,"end":40,"cssClass":"pl-s1"},{"start":41,"end":46,"cssClass":"pl-s1"},{"start":46,"end":47,"cssClass":"pl-kos"}],[{"start":4,"end":11,"cssClass":"pl-s1"},{"start":12,"end":17,"cssClass":"pl-s1"},{"start":18,"end":44,"cssClass":"pl-en"}],[{"start":4,"end":11,"cssClass":"pl-s1"},{"start":12,"end":24,"cssClass":"pl-en"},{"start":25,"end":55,"cssClass":"pl-v"},{"start":56,"end":63,"cssClass":"pl-s1"},{"start":65,"end":75,"cssClass":"pl-s1"},{"start":75,"end":76,"cssClass":"pl-c1"},{"start":76,"end":89,"cssClass":"pl-s1"},{"start":90,"end":100,"cssClass":"pl-s1"}],[],[{"start":4,"end":20,"cssClass":"pl-c"}],[{"start":4,"end":11,"cssClass":"pl-s1"},{"start":12,"end":17,"cssClass":"pl-en"}],[],[{"start":4,"end":32,"cssClass":"pl-c"}],[{"start":4,"end":11,"cssClass":"pl-s1"},{"start":12,"end":23,"cssClass":"pl-s1"},{"start":24,"end":41,"cssClass":"pl-en"}],[{"start":4,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":17,"end":24,"cssClass":"pl-s1"},{"start":25,"end":36,"cssClass":"pl-s1"},{"start":37,"end":51,"cssClass":"pl-en"},{"start":52,"end":59,"cssClass":"pl-s1"},{"start":60,"end":69,"cssClass":"pl-s1"}],[{"start":4,"end":19,"cssClass":"pl-s1"},{"start":20,"end":21,"cssClass":"pl-c1"},{"start":22,"end":29,"cssClass":"pl-s1"},{"start":30,"end":41,"cssClass":"pl-s1"},{"start":42,"end":54,"cssClass":"pl-en"},{"start":55,"end":62,"cssClass":"pl-s1"},{"start":63,"end":72,"cssClass":"pl-s1"}],[{"start":4,"end":6,"cssClass":"pl-k"},{"start":7,"end":14,"cssClass":"pl-s1"},{"start":15,"end":26,"cssClass":"pl-s1"},{"start":27,"end":42,"cssClass":"pl-s1"}],[{"start":8,"end":23,"cssClass":"pl-s1"},{"start":24,"end":39,"cssClass":"pl-en"},{"start":40,"end":53,"cssClass":"pl-s1"},{"start":54,"end":64,"cssClass":"pl-s1"},{"start":66,"end":76,"cssClass":"pl-s1"},{"start":76,"end":77,"cssClass":"pl-c1"},{"start":77,"end":87,"cssClass":"pl-s1"}],[{"start":4,"end":11,"cssClass":"pl-s1"},{"start":12,"end":23,"cssClass":"pl-s1"},{"start":24,"end":41,"cssClass":"pl-en"}],[],[{"start":4,"end":30,"cssClass":"pl-c"}],[{"start":4,"end":42,"cssClass":"pl-c"}],[{"start":4,"end":6,"cssClass":"pl-k"},{"start":7,"end":14,"cssClass":"pl-s1"},{"start":15,"end":19,"cssClass":"pl-s1"},{"start":20,"end":33,"cssClass":"pl-s1"},{"start":34,"end":36,"cssClass":"pl-c1"},{"start":37,"end":38,"cssClass":"pl-c1"}],[{"start":8,"end":10,"cssClass":"pl-k"},{"start":11,"end":22,"cssClass":"pl-s1"},{"start":23,"end":37,"cssClass":"pl-s1"}],[{"start":12,"end":60,"cssClass":"pl-c"}],[{"start":12,"end":30,"cssClass":"pl-c"}],[{"start":12,"end":19,"cssClass":"pl-s1"},{"start":20,"end":25,"cssClass":"pl-s1"},{"start":26,"end":41,"cssClass":"pl-en"},{"start":42,"end":55,"cssClass":"pl-s1"},{"start":56,"end":66,"cssClass":"pl-s1"},{"start":68,"end":86,"cssClass":"pl-s1"},{"start":86,"end":87,"cssClass":"pl-c1"},{"start":87,"end":92,"cssClass":"pl-c1"}],[{"start":12,"end":26,"cssClass":"pl-c"}],[{"start":12,"end":15,"cssClass":"pl-k"},{"start":16,"end":21,"cssClass":"pl-s1"}],[{"start":12,"end":15,"cssClass":"pl-k"},{"start":16,"end":23,"cssClass":"pl-s1"}],[{"start":12,"end":17,"cssClass":"pl-s1"},{"start":18,"end":22,"cssClass":"pl-s1"},{"start":23,"end":34,"cssClass":"pl-en"}],[],[{"start":12,"end":16,"cssClass":"pl-k"},{"start":17,"end":21,"cssClass":"pl-s1"},{"start":22,"end":28,"cssClass":"pl-k"},{"start":29,"end":53,"cssClass":"pl-v"}],[],[{"start":12,"end":37,"cssClass":"pl-c"}],[{"start":12,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":44,"cssClass":"pl-v"},{"start":45,"end":60,"cssClass":"pl-en"}],[{"start":16,"end":29,"cssClass":"pl-s1"},{"start":30,"end":40,"cssClass":"pl-s1"}],[{"start":16,"end":33,"cssClass":"pl-s1"},{"start":33,"end":34,"cssClass":"pl-c1"},{"start":34,"end":38,"cssClass":"pl-c1"}],[{"start":16,"end":27,"cssClass":"pl-s1"},{"start":27,"end":28,"cssClass":"pl-c1"},{"start":28,"end":33,"cssClass":"pl-s1"},{"start":34,"end":41,"cssClass":"pl-s1"}],[],[{"start":12,"end":48,"cssClass":"pl-c"}],[{"start":12,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":25,"cssClass":"pl-s1"},{"start":26,"end":42,"cssClass":"pl-en"}],[{"start":12,"end":17,"cssClass":"pl-s1"},{"start":18,"end":33,"cssClass":"pl-en"}],[{"start":16,"end":29,"cssClass":"pl-s1"},{"start":30,"end":40,"cssClass":"pl-s1"},{"start":42,"end":60,"cssClass":"pl-s1"},{"start":60,"end":61,"cssClass":"pl-c1"},{"start":61,"end":65,"cssClass":"pl-c1"},{"start":67,"end":81,"cssClass":"pl-s1"},{"start":81,"end":82,"cssClass":"pl-c1"},{"start":82,"end":87,"cssClass":"pl-s"}],[],[{"start":8,"end":12,"cssClass":"pl-k"}],[{"start":12,"end":19,"cssClass":"pl-s1"},{"start":20,"end":25,"cssClass":"pl-s1"},{"start":26,"end":41,"cssClass":"pl-en"}],[{"start":16,"end":29,"cssClass":"pl-s1"},{"start":30,"end":40,"cssClass":"pl-s1"},{"start":42,"end":60,"cssClass":"pl-s1"},{"start":60,"end":61,"cssClass":"pl-c1"},{"start":61,"end":65,"cssClass":"pl-c1"}],[],[],[{"start":8,"end":25,"cssClass":"pl-c"}],[{"start":8,"end":17,"cssClass":"pl-s1"},{"start":18,"end":33,"cssClass":"pl-en"},{"start":34,"end":47,"cssClass":"pl-s1"},{"start":48,"end":58,"cssClass":"pl-s1"}],[],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":8,"cssClass":"pl-en"}],[{"start":4,"end":10,"cssClass":"pl-s1"},{"start":11,"end":12,"cssClass":"pl-c1"},{"start":13,"end":29,"cssClass":"pl-v"},{"start":31,"end":46,"cssClass":"pl-v"},{"start":47,"end":64,"cssClass":"pl-v"}],[{"start":4,"end":15,"cssClass":"pl-s1"},{"start":17,"end":30,"cssClass":"pl-s1"},{"start":31,"end":32,"cssClass":"pl-c1"},{"start":33,"end":39,"cssClass":"pl-s1"},{"start":40,"end":67,"cssClass":"pl-en"}],[{"start":4,"end":15,"cssClass":"pl-s1"},{"start":16,"end":17,"cssClass":"pl-c1"},{"start":18,"end":22,"cssClass":"pl-en"},{"start":23,"end":38,"cssClass":"pl-v"},{"start":40,"end":51,"cssClass":"pl-s1"}],[{"start":4,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":24,"cssClass":"pl-en"},{"start":25,"end":42,"cssClass":"pl-v"},{"start":44,"end":57,"cssClass":"pl-s1"}],[],[{"start":4,"end":21,"cssClass":"pl-en"},{"start":22,"end":33,"cssClass":"pl-s1"},{"start":35,"end":48,"cssClass":"pl-s1"}],[],[],[{"start":0,"end":2,"cssClass":"pl-k"},{"start":3,"end":11,"cssClass":"pl-s1"},{"start":12,"end":14,"cssClass":"pl-c1"},{"start":15,"end":25,"cssClass":"pl-s"}],[{"start":4,"end":8,"cssClass":"pl-en"}]],"csv":null,"csvError":null,"dependabotInfo":{"showConfigurationBanner":false,"configFilePath":null,"networkDependabotPath":"/philschmid/deep-learning-pytorch-huggingface/network/updates","dismissConfigurationNoticePath":"/settings/dismiss-notice/dependabot_configuration_notice","configurationNoticeDismissed":null,"repoAlertsPath":"/philschmid/deep-learning-pytorch-huggingface/security/dependabot","repoSecurityAndAnalysisPath":"/philschmid/deep-learning-pytorch-huggingface/settings/security_analysis","repoOwnerIsOrg":false,"currentUserCanAdminRepo":false},"displayName":"run_ds_lora.py","displayUrl":"https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/run_ds_lora.py?raw=true","headerInfo":{"blobSize":"3.85 KB","deleteInfo":{"deleteTooltip":"You must be signed in to make or propose changes"},"editInfo":{"editTooltip":"You must be signed in to make or propose changes"},"ghDesktopPath":"https://desktop.github.com","gitLfsPath":null,"onBranch":true,"shortPath":"2c2f1ed","siteNavLoginPath":"/login?return_to=https%3A%2F%2Fgithub.com%2Fphilschmid%2Fdeep-learning-pytorch-huggingface%2Fblob%2Fmain%2Ftraining%2Frun_ds_lora.py","isCSV":false,"isRichtext":false,"toc":null,"lineInfo":{"truncatedLoc":"113","truncatedSloc":"92"},"mode":"file"},"image":false,"isCodeownersFile":null,"isPlain":false,"isValidLegacyIssueTemplate":false,"issueTemplateHelpUrl":"https://docs.github.com/articles/about-issue-and-pull-request-templates","issueTemplate":null,"discussionTemplate":null,"language":"Python","languageID":303,"large":false,"loggedIn":false,"newDiscussionPath":"/philschmid/deep-learning-pytorch-huggingface/discussions/new","newIssuePath":"/philschmid/deep-learning-pytorch-huggingface/issues/new","planSupportInfo":{"repoIsFork":null,"repoOwnedByCurrentUser":null,"requestFullPath":"/philschmid/deep-learning-pytorch-huggingface/blob/main/training/run_ds_lora.py","showFreeOrgGatedFeatureMessage":null,"showPlanSupportBanner":null,"upgradeDataAttributes":null,"upgradePath":null},"publishBannersInfo":{"dismissActionNoticePath":"/settings/dismiss-notice/publish_action_from_dockerfile","dismissStackNoticePath":"/settings/dismiss-notice/publish_stack_from_file","releasePath":"/philschmid/deep-learning-pytorch-huggingface/releases/new?marketplace=true","showPublishActionBanner":false,"showPublishStackBanner":false},"renderImageOrRaw":false,"richText":null,"renderedFileInfo":null,"shortPath":null,"tabSize":8,"topBannersInfo":{"overridingGlobalFundingFile":false,"globalPreferredFundingPath":null,"repoOwner":"philschmid","repoName":"deep-learning-pytorch-huggingface","showInvalidCitationWarning":false,"citationHelpUrl":"https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files","showDependabotConfigurationBanner":false,"actionsOnboardingTip":null},"truncated":false,"viewable":true,"workflowRedirectUrl":null,"symbols":{"timedOut":false,"notAnalyzed":false,"symbols":[{"name":"ScriptArguments","kind":"class","identStart":377,"identEnd":392,"extentStart":371,"extentEnd":1243,"fullyQualifiedName":"ScriptArguments","identUtf16":{"start":{"lineNumber":15,"utf16Col":6},"end":{"lineNumber":15,"utf16Col":21}},"extentUtf16":{"start":{"lineNumber":15,"utf16Col":0},"end":{"lineNumber":38,"utf16Col":5}}},{"name":"model_id","kind":"constant","identStart":494,"identEnd":502,"extentStart":494,"extentEnd":663,"fullyQualifiedName":"ScriptArguments.model_id","identUtf16":{"start":{"lineNumber":19,"utf16Col":4},"end":{"lineNumber":19,"utf16Col":12}},"extentUtf16":{"start":{"lineNumber":19,"utf16Col":4},"end":{"lineNumber":23,"utf16Col":5}}},{"name":"dataset_path","kind":"constant","identStart":668,"identEnd":680,"extentStart":668,"extentEnd":824,"fullyQualifiedName":"ScriptArguments.dataset_path","identUtf16":{"start":{"lineNumber":24,"utf16Col":4},"end":{"lineNumber":24,"utf16Col":16}},"extentUtf16":{"start":{"lineNumber":24,"utf16Col":4},"end":{"lineNumber":27,"utf16Col":5}}},{"name":"lora_alpha","kind":"constant","identStart":829,"identEnd":839,"extentStart":829,"extentEnd":874,"fullyQualifiedName":"ScriptArguments.lora_alpha","identUtf16":{"start":{"lineNumber":28,"utf16Col":4},"end":{"lineNumber":28,"utf16Col":14}},"extentUtf16":{"start":{"lineNumber":28,"utf16Col":4},"end":{"lineNumber":28,"utf16Col":49}}},{"name":"lora_dropout","kind":"constant","identStart":879,"identEnd":891,"extentStart":879,"extentEnd":929,"fullyQualifiedName":"ScriptArguments.lora_dropout","identUtf16":{"start":{"lineNumber":29,"utf16Col":4},"end":{"lineNumber":29,"utf16Col":16}},"extentUtf16":{"start":{"lineNumber":29,"utf16Col":4},"end":{"lineNumber":29,"utf16Col":54}}},{"name":"lora_r","kind":"constant","identStart":934,"identEnd":940,"extentStart":934,"extentEnd":975,"fullyQualifiedName":"ScriptArguments.lora_r","identUtf16":{"start":{"lineNumber":30,"utf16Col":4},"end":{"lineNumber":30,"utf16Col":10}},"extentUtf16":{"start":{"lineNumber":30,"utf16Col":4},"end":{"lineNumber":30,"utf16Col":45}}},{"name":"use_flash_attn","kind":"constant","identStart":980,"identEnd":994,"extentStart":980,"extentEnd":1116,"fullyQualifiedName":"ScriptArguments.use_flash_attn","identUtf16":{"start":{"lineNumber":31,"utf16Col":4},"end":{"lineNumber":31,"utf16Col":18}},"extentUtf16":{"start":{"lineNumber":31,"utf16Col":4},"end":{"lineNumber":34,"utf16Col":5}}},{"name":"merge_adapters","kind":"constant","identStart":1121,"identEnd":1135,"extentStart":1121,"extentEnd":1243,"fullyQualifiedName":"ScriptArguments.merge_adapters","identUtf16":{"start":{"lineNumber":35,"utf16Col":4},"end":{"lineNumber":35,"utf16Col":18}},"extentUtf16":{"start":{"lineNumber":35,"utf16Col":4},"end":{"lineNumber":38,"utf16Col":5}}},{"name":"training_function","kind":"function","identStart":1250,"identEnd":1267,"extentStart":1246,"extentEnd":3583,"fullyQualifiedName":"training_function","identUtf16":{"start":{"lineNumber":41,"utf16Col":4},"end":{"lineNumber":41,"utf16Col":21}},"extentUtf16":{"start":{"lineNumber":41,"utf16Col":0},"end":{"lineNumber":99,"utf16Col":59}}},{"name":"main","kind":"function","identStart":3590,"identEnd":3594,"extentStart":3586,"extentEnd":3901,"fullyQualifiedName":"main","identUtf16":{"start":{"lineNumber":102,"utf16Col":4},"end":{"lineNumber":102,"utf16Col":8}},"extentUtf16":{"start":{"lineNumber":102,"utf16Col":0},"end":{"lineNumber":108,"utf16Col":49}}}]}},"copilotInfo":null,"csrf_tokens":{"/philschmid/deep-learning-pytorch-huggingface/branches":{"post":"Zt4Gln000BHy39jBWJY6QZT_RHLKUF81rL0vB_CUtBnHgB5CqXj3JunRoQc-gmqQ_Jli6CJK3QOQgr8bttIV-Q"},"/repos/preferences":{"post":"3sq1LQvuvttiPhHS3xkKeHNEWSRd1D6st2G9vT1lYpL_E0NjsGdmyjlKTWAv4fmdLmrxo-bXqlDo0aRXETV6tw"}}},"title":"deep-learning-pytorch-huggingface/training/run_ds_lora.py at main · philschmid/deep-learning-pytorch-huggingface"}