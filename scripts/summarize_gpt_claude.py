"""
GPT-4o vs Claude Sonnet 4.6 실험 결과 종합 (Gemini 제외)
"""
import json
import sys
from pathlib import Path

# UTF-8 출력 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    results_dir = Path('results/cross_domain')
    domains = ['medical', 'legal', 'code', 'logic', 'long_context']
    models = ['gpt_4o', 'claude_sonnet_4_6']
    agents = ['1', '2', '3', '4']

    domain_sizes = {
        'medical': 5000,
        'legal': 3000,
        'code': 1000,
        'logic': 1500,
        'long_context': 902
    }

    print('='*120)
    print('GPT-4o vs Claude Sonnet 4.6 실험 결과 종합')
    print('='*120)
    print()

    for domain in domains:
        expected = domain_sizes[domain]
        print(f'\n{"="*120}')
        print(f'{domain.upper()} 도메인 (예상: {expected}문제)')
        print('='*120)
        print(f'{"모델":<20s} | {"1-agent":<20s} | {"2-agent":<20s} | {"3-agent":<20s} | {"4-agent":<20s}')
        print('-'*120)

        for model in models:
            row = [model]

            for agent in agents:
                file_name = f"{domain}_{model}_{agent}agent.json"
                parallel_file_name = f"{domain}_{model}_{agent}agent_parallel.json"

                candidates = [
                    results_dir / 'merged' / parallel_file_name,
                    results_dir / 'merged' / file_name,
                    results_dir / file_name,
                ]

                stats = None
                for file_path in candidates:
                    if file_path.exists():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            if isinstance(data, dict):
                                if 'results' in data:
                                    total = data.get('total_questions', len(data['results']))
                                    correct = data.get('correct', sum(1 for item in data['results'] if item.get('is_correct', False)))
                                    accuracy = data.get('accuracy', (correct / total * 100) if total > 0 else 0)
                                else:
                                    continue
                            elif isinstance(data, list):
                                total = len(data)
                                correct = sum(1 for item in data if item.get('is_correct', False))
                                accuracy = (correct / total * 100) if total > 0 else 0
                            else:
                                continue

                            stats = {'total': total, 'correct': correct, 'accuracy': accuracy}
                            break
                        except:
                            continue

                if stats and stats['total'] > 0:
                    total = stats['total']
                    correct = stats['correct']
                    accuracy = stats['accuracy']

                    if total == expected:
                        row.append(f'{correct}/{total} ({accuracy:.1f}%)')
                    else:
                        pct = (total / expected * 100) if expected > 0 else 0
                        row.append(f'⚠️ {correct}/{total} ({pct:.0f}%)')
                else:
                    row.append('❌ 미완료')

            print(f'{row[0]:<20s} | {row[1]:<20s} | {row[2]:<20s} | {row[3]:<20s} | {row[4]:<20s}')

    print('\n\n' + '='*120)
    print('도메인별 완료 상태 요약')
    print('='*120)
    print(f'{"도메인":<15s} | {"GPT-4o":<30s} | {"Claude Sonnet 4.6":<30s}')
    print('-'*120)

    gpt_total = 0
    gpt_completed = 0
    claude_total = 0
    claude_completed = 0

    for domain in domains:
        expected = domain_sizes[domain]
        row = [domain]

        for model in models:
            completed = 0

            for agent in agents:
                file_name = f"{domain}_{model}_{agent}agent.json"
                parallel_file_name = f"{domain}_{model}_{agent}agent_parallel.json"

                candidates = [
                    results_dir / 'merged' / parallel_file_name,
                    results_dir / 'merged' / file_name,
                    results_dir / file_name,
                ]

                for file_path in candidates:
                    if file_path.exists():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            if isinstance(data, dict) and 'results' in data:
                                total = data.get('total_questions', len(data['results']))
                            elif isinstance(data, list):
                                total = len(data)
                            else:
                                continue

                            if total == expected:
                                completed += 1
                            break
                        except:
                            continue

            if model == 'gpt_4o':
                gpt_total += 4
                gpt_completed += completed
            else:
                claude_total += 4
                claude_completed += completed

            if completed == 4:
                status = f'✅ {completed}/4 완료'
            elif completed > 0:
                status = f'⚠️ {completed}/4 부분완료'
            else:
                status = f'❌ {completed}/4 미완료'

            row.append(status)

        print(f'{row[0]:<15s} | {row[1]:<30s} | {row[2]:<30s}')

    print('='*120)
    print(f'\n전체 진행률:')
    print(f'  GPT-4o: {gpt_completed}/{gpt_total} 완료 ({gpt_completed/gpt_total*100:.1f}%)')
    print(f'  Claude Sonnet 4.6: {claude_completed}/{claude_total} 완료 ({claude_completed/claude_total*100:.1f}%)')
    print(f'  전체: {gpt_completed+claude_completed}/{gpt_total+claude_total} 완료 ({(gpt_completed+claude_completed)/(gpt_total+claude_total)*100:.1f}%)')
    print('='*120)

if __name__ == '__main__':
    main()
