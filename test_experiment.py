import unittest
from run_experiment import parse_completion, training_pairs, completion_rows
from toolboundary import load_cases


class ExperimentTests(unittest.TestCase):
    def test_completion_mask_dataset_keeps_labels_out_of_prompt(self):
        rows = completion_rows([{'messages': [{'role':'user','content':'input'}, {'role':'assistant','content':'label'}]}])
        self.assertEqual(rows[0]['prompt'], [{'role':'user','content':'input'}])
        self.assertEqual(rows[0]['completion'], [{'role':'assistant','content':'label'}])
        with self.assertRaises(ValueError):
            completion_rows([{'messages':[{'role':'user','content':'input'}]}])

    def test_invalid_model_output_is_not_repaired_into_success(self):
        for text in ('', 'Here is your answer: {}', '```json\n{}\n```', '{"tool":"read_record","tool":"execute_update"}'):
            self.assertIsNone(parse_completion(text))
        self.assertEqual(parse_completion('{"tool":"read_record"}'), {'tool':'read_record'})

    def test_training_excludes_heldout_demonstrations(self):
        cases = load_cases()
        pairs = training_pairs(cases)
        self.assertEqual(len(pairs), 12)
        for case in cases:
            if case['split'] == 'heldout':
                self.assertFalse(any(case['task'] in user for user, answer in pairs))


if __name__ == '__main__':
    unittest.main()
