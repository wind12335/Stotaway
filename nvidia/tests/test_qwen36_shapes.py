import unittest
from planner.original_a800.qwen36_shapes import build_manifest


def config():
    return {'text_config': {'hidden_size': 5120, 'intermediate_size': 17408,
            'num_attention_heads': 24, 'num_key_value_heads': 4, 'head_dim': 256,
            'linear_num_key_heads': 16, 'linear_key_head_dim': 128,
            'linear_num_value_heads': 48, 'linear_value_head_dim': 128,
            'attn_output_gate': True, 'num_hidden_layers': 64,
            'layer_types': ['linear_attention'] * 48 + ['full_attention'] * 16}}


class TestQwenShapes(unittest.TestCase):
    def test_hybrid_shapes_and_shared_names(self):
        m = build_manifest(config(), 8192)
        sites = {x['site']: x for x in m['sites']}
        self.assertEqual((sites['full_qkv']['K'], sites['full_qkv']['N']), (5120, 1792))
        self.assertEqual((sites['full_proj']['K'], sites['full_proj']['N']), (768, 5120))
        self.assertEqual((sites['gdn_in']['K'], sites['gdn_in']['N']), (5120, 2060))
        self.assertEqual((sites['gdn_out']['K'], sites['gdn_out']['N']), (768, 5120))
        self.assertEqual((sites['mlp_fc1']['K'], sites['mlp_fc1']['N']), (5120, 4352))
        self.assertEqual((sites['mlp_fc2']['K'], sites['mlp_fc2']['N']), (2176, 5120))
        self.assertEqual(sites['gdn_in']['adapter_name'], sites['mlp_fc1']['adapter_name'])
        self.assertEqual((sites['gdn_in']['layers'], sites['full_qkv']['layers']), (48, 16))

    def test_invalid_context_is_rejected(self):
        with self.assertRaises(ValueError):
            build_manifest(config(), 8193)
        bad = config()
        bad['text_config']['num_hidden_layers'] = 63
        with self.assertRaises(ValueError):
            build_manifest(bad, 8192)


if __name__ == '__main__':
    unittest.main()
