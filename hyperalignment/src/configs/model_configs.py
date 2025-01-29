from types import SimpleNamespace


model_configs = SimpleNamespace(**{})

"""
text_encoder = 384/768

"""

# image encoders
model_configs.image_encoder_configs = {
	768: [
    	"vit_base_patch16_224",
		"deit_base_patch16_224",
		"swin_small_patch4_window7_224.ms_in22k_ft_in1k",
		"swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
		"convnext_small.fb_in22k_ft_in1k",
		"beit_base_patch16_224.in22k_ft_in22k_in1k",
		"eva02_base_patch14_224.mim_in22k",
	],
	384: [
		"vit_small_patch16_224",
		"deit_small_patch16_224",
		"eva02_small_patch14_224.mim_in22k",
		"visformer_tiny.in1k"
	],

	1024: [
		"swin_base_patch4_window7_224.ms_in22k_ft_in1k",
		"vit_large_patch16_224",
		"convnext_base.fb_in22k_ft_in1k",
		"vit_large_patch14_clip_336.laion2b_ft_in12k_in1k"
	]
}

# text encoders
model_configs.text_encoder_configs = {
	768: [
		"sentence-t5-base",
		# "gtr-t5-large",
		# "all-mpnet-base-v2",
		# "all-mpnet-base-v1",
		# "paraphrase-mpnet-base-v2",
	],

	384: [
		"all-MiniLM-L12-v2",
		"all-MiniLM-L12-v1",
		"all-MiniLM-L6-v2",
		"all-MiniLM-L6-v1",
		"paraphrase-MiniLM-L12-v2",
		"paraphrase-MiniLM-L6-v2",
		"multi-qa-MiniLM-L6-cos-v1"
	],

	1024: [
		"all-roberta-large-v1",
	]
}

model_configs.ID_vit_recipe_variant_configs = {
    768 : [
        "vit_base_patch16_224",
        "vit_base_patch16_224.augreg_in1k",
        "vit_base_patch16_224.mae",
        "vit_base_patch16_224.dino",
        "vit_base_patch16_clip_224.openai",
        "vit_base_patch16_rope_reg1_gap_256.sbb_in1k",
    ]
}

model_configs.ID_ie_family_variant_configs = {
	768: [
    	"vit_base_patch16_224",
		"deit_base_patch16_224",
		"swin_small_patch4_window7_224.ms_in22k_ft_in1k",
		"convnext_small.fb_in22k_ft_in1k",
		"beit_base_patch16_224.in22k_ft_in22k_in1k",
		"eva02_base_patch14_224.mim_in22k",
	],
}

model_configs.ID_multi_mapper_configs = {
    # 768: ['vit_base_patch16_224', 'vit_base_patch16_224.augreg_in1k', 'vit_base_patch16_224.dino', 'vit_base_patch16_clip_224.openai'],
    # 384: ['vit_small_patch16_224', 'deit_small_patch16_224', 'eva02_small_patch14_224.mim_in22k', 'visformer_tiny.in1k']
    384: [
		"vit_small_patch16_224",
		"deit_small_patch16_224",
		"deit3_small_patch16_224.fb_in1k",
		"deit3_small_patch16_224.fb_in22k_ft_in1k",
		"efficientvit_m5.r224_in1k",
		"flexivit_small.300ep_in1k",
		"visformer_tiny.in1k",
		"volo_d1_224.sail_in1k",
		"xcit_small_12_p8_224.fb_in1k", ## - isolated checked, repeated unchecked
		"eva02_small_patch14_224.mim_in22k",
	],
    768: [
		"vit_base_patch16_224",
		"vit_base_patch32_224.augreg_in21k_ft_in1k",
		"vit_base_patch32_clip_224.laion2b_ft_in12k_in1k",
		"deit_base_patch16_224",
		"deit3_base_patch16_224.fb_in22k_ft_in1k",
		"beit_base_patch16_224.in22k_ft_in22k_in1k",
		"swin_small_patch4_window7_224.ms_in22k_ft_in1k",
		"convnext_small.fb_in22k_ft_in1k",
		"volo_d4_224.sail_in1k", ## - iso
		"maxvit_base_tf_224.in1k",
	],
	1024: [
        "vit_large_patch16_224",
        "vit_large_patch16_224.augreg_in21k_ft_in1k",
		"vit_large_patch14_clip_336.laion2b_ft_in12k_in1k", #
		"deit3_large_patch16_384.fb_in22k_ft_in1k", #
		"eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", #
		"beit_large_patch16_384.in22k_ft_in22k_in1k", #
		"beitv2_large_patch16_224.in1k_ft_in22k_in1k", #
		"swin_base_patch4_window7_224.ms_in22k_ft_in1k", #
		"convnext_base.fb_in22k_ft_in1k", # 
		"convnextv2_base.fcmae_ft_in22k_in1k" #
	]
}
# "maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k", #
# "maxvit_large_tf_384.in21k_ft_in1k", ##

model_configs.ID_experiment_configs = {
    "id_vitr": {
        "image_encoders": model_configs.ID_vit_recipe_variant_configs[768],
        "text_encoders": model_configs.text_encoder_configs[768],
        "extensions": ["vanilla", "augreg", "mae", "dino", "openai", "rope"]
	},
    "id_famv": {
        "image_encoders": model_configs.ID_ie_family_variant_configs[768],
        "text_encoders": model_configs.text_encoder_configs[768],
        "extensions": ["vit", "deit", "swin", "convnext", "beit", "eva-02"]
	},
	"multi_distil": {
        "image_encoders": model_configs.image_encoder_configs[384],
        "text_encoders": model_configs.text_encoder_configs[768],
        "extensions": ["vit", "deit", "eva", "visformer"]
    },
    "large_emb": {
        "image_encoders": model_configs.image_encoder_configs[1024],
        "text_encoders": model_configs.text_encoder_configs[768],
        "extensions": ["swin_base", "vit_large", "convnext_base", "vit_large_clip"]
	},
    "multi_mapper": {
        1024: {
            "image_encoders": model_configs.ID_multi_mapper_configs[1024],
            "text_encoders": model_configs.text_encoder_configs[768]
        },
        768: {
            "image_encoders": model_configs.ID_multi_mapper_configs[768],
            "text_encoders": model_configs.text_encoder_configs[768]
		},
        384: {
            "image_encoders": model_configs.ID_multi_mapper_configs[384],
            "text_encoders": model_configs.text_encoder_configs[768]
		}
	}
}

model_configs.icml_encoder_configs = {
    "image_encoders": model_configs.ID_multi_mapper_configs,
    "text_encoders": {
        384: ["all-MiniLM-L12-v2"],
        768: ["all-mpnet-base-v1"],
        1024: ["all-roberta-large-v1"]
	}
}

model_configs.hnet_decoder_configs = {
    "mlp": {
        "decoder_type": "mlp",
        "hidden_layer_factors": [4, 16],
	},
    "chunked_mlp": {
        "decoder_type": "chunked_mlp",
        "chunk_dim": 256,
        "hidden_layer_factors": [4, 16],
	},
    "feather_map": {
        "decoder_type": "feather_map",
        "rank": 32,
        "hidden_layer_factors": [4, 16],
	},
}
