"""Interactive local case explorer. Only in-memory synthetic records are changed."""
import json
from pathlib import Path
from toolboundary import evaluate, load_cases, replay


def main():
    cases = load_cases()
    print('ToolBoundary | Local authorization lab')
    print('Synthetic data only. No model trained; no external tools execute.\n')
    while True:
        for number, case in enumerate(cases, 1):
            print(f'{number:2}. {case["id"]}')
        choice = input('\nChoose a case (1-24), score a prediction file (s), or quit (q): ').strip()
        if choice.lower() == 'q':
            return
        if choice.lower() == 's':
            path = Path(input('Prediction JSON file: ').strip().strip('"'))
            try:
                predictions = json.loads(path.read_text(encoding='utf-8'))
                result = evaluate(cases, predictions, input('Actual model/run identity: ').strip() or 'Unverified identity')
                output = path.with_name(path.stem + '-evaluation.json')
                if output.exists():
                    print('Output already exists; rename your input to preserve previous evidence.')
                    continue
                output.write_text(json.dumps(result, indent=2), encoding='utf-8')
                print(json.dumps(result['metrics'], indent=2))
                print(f'Audit saved: {output}')
            except (ValueError, OSError) as error:
                print(f'Not evaluated: {error}')
            continue
        try:
            number = int(choice)
            if not 1 <= number <= len(cases):
                raise ValueError('Choose a listed case')
            case = cases[number - 1]
        except ValueError as error:
            print(error)
            continue
        print('\nTask:', case['task'])
        print('Trusted grants:', json.dumps(case['grants']))
        print('Records:', json.dumps(case['records']))
        print('Untrusted document:', case['document'])
        print('\nTools: read_record, propose_update, execute_update, request_approval')
        tool = input('Tool: ').strip()
        record = input(f'Record [{case["target"]}]: ').strip() or case['target']
        version = input(f'Version [{case["records"][case["target"]]["version"]}]: ').strip()
        try:
            version = int(version) if version else case['records'][case['target']]['version']
        except ValueError:
            print('Invalid version; nothing executed.')
            continue
        value = input('New value (empty for read/approval): ')
        action = dict(tool=tool, record=record, version=version, value=value)
        result = replay(case, action)
        print('\nDecision:', 'ALLOWED' if result['allowed'] else 'BLOCKED')
        print('Reason:', result['reason'])
        print('Before:', json.dumps(result['before']))
        print('After:', json.dumps(result['after']))
        print('Expected single-step action:', json.dumps(case['expected']))
        print('Exact task match:', action == case['expected'])
        input('\nPress Enter to choose another case...')


if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print('\nClosed; no external records changed.')
