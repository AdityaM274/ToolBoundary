import unittest
from build_cases import build
from toolboundary import guard, prompt
from run_experiment import training_pairs


class CorpusTests(unittest.TestCase):
    def test_labels_match_policy_across_all_cases(self):
        rows = build('train-v2', 240, 'development') + build('held-v2', 24, 'heldout')
        for case in rows:
            self.assertTrue(guard(case, case['expected'])[0], case['id'])
            attempted = dict(case['expected'], tool=case['task'].split('requested operation is ')[1].split('.')[0]) if case['split']=='heldout' else None
            if attempted is not None and case['expected']['tool'] == 'request_approval':
                if attempted['tool'] == 'execute_update':
                    attempted['value'] = ['ready','reviewed','closed','pending'][int(case['id'].split('-')[-1])%4]
                self.assertFalse(guard(case, attempted)[0], case['id'])

    def test_independent_ids_and_prompts(self):
        train = build('train-v2', 240, 'development')
        evaluation = build('held-v2', 24, 'heldout')
        self.assertFalse({c['id'] for c in train} & {c['id'] for c in evaluation})
        pairs = training_pairs(train)
        for case in evaluation:
            self.assertFalse(any(case['task'] in user for user, answer in pairs))
        self.assertEqual({c['expected']['tool'] for c in evaluation}, {'read_record','propose_update','execute_update','request_approval'})


if __name__ == '__main__':
    unittest.main()
