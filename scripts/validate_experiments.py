"""
실험 결과 검증 스크립트 - 모든 모델/에이전트가 동일한 문제 수를 평가했는지 확인
"""
import json
from pathlib import Path
from collections import defaultdict
import sys

# UTF-8 출력 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_experiment_stats(file_path):
    """실험 결과 파일에서 통계 추출"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            if 'results' in data:
                total = data.get('total_questions', len(data['results']))
                correct = data.get('correct', sum(1 for item in data['results'] if item.get('is_correct', False)))
                accuracy = data.get('accuracy', (correct / total * 100) if total > 0 else 0)
            else:
                return None
        elif isinstance(data, list):
            total = len(data)
            correct = sum(1 for item in data if item.get('is_correct', False))
            accuracy = (correct / total * 100) if total > 0 else 0
        else:
            return None

        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }
    except Exception:
        return None

def main():
    results_dir = Path('results/cross_domain')

    domains = ['medical', 'legal', 'code', 'logic', 'long_context']
    models = ['gpt_4o', 'gemini_2_5_pro', 'claude_sonnet_4_6']
    agents = ['1', '2', '3', '4']

    # 도메인별 예상 질문 수
    domain_sizes = {
        'medical': 5000,
        'legal': 3000,
        'code': 1000,
        'logic': 1500,
        'long_context': 902
    }

    print('='*120)
    print('실험 결과 검증 - 문제 수 기준')
    print('='*120)
    print()

    # 도메인별로 확인
    for domain in domains:
        expected = domain_sizes[domain]
        print(f'\n{'='*120}')
        print(f'{domain.upper()} 도메인 (예상: {expected}문제)')
        print('='*120)
        print(f'{"모델":<20s} | {"1-agent":<15s} | {"2-agent":<15s} | {"3-agent":<15s} | {"4-agent":<15s} | 상태')
        print('-'*120)

        for model in models:
            row = [model]
            has_valid = False
            all_match = True

            for agent in agents:
                # 여러 파일 위치 확인
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
                        stats = get_experiment_stats(file_path)
                        if stats and stats['total'] > 0:
                            break

                if stats and stats['total'] > 0:
                    total = stats['total']
                    correct = stats['correct']
                    accuracy = stats['accuracy']

                    # 문제 수가 예상과 일치하는지 확인
                    if total == expected:
                        row.append(f'{correct}/{total} ({accuracy:.1f}%)')
                        has_valid = True
                    else:
                        # 테스트 모드 또는 불일치
                        pct = (total / expected * 100) if expected > 0 else 0
                        row.append(f'⚠️ {total}문제 ({pct:.0f}%)')
                        all_match = False
                        if pct >= 50:  # 50% 이상은 유효한 데이터로 간주
                            has_valid = True
                else:
                    row.append('❌ 미완료')
                    all_match = False

            # 상태 표시
            if all_match and has_valid:
                status = '✅ 완료'
            elif has_valid:
                status = '⚠️ 부분완료'
            else:
                status = '❌ 미완료'

            row.append(status)
            print(f'{row[0]:<20s} | {row[1]:<15s} | {row[2]:<15s} | {row[3]:<15s} | {row[4]:<15s} | {row[5]}')

    # 요약
    print(f'\n\n{'='*120}')
    print('실험 상태 요약 (전체 문제 수 기준)')
    print('='*120)
    print(f'{"도메인":<15s} | {"GPT-4o":<25s} | {"Gemini 2.5 Pro":<25s} | {"Claude Sonnet 4.6":<25s}')
    print('-'*120)

    for domain in domains:
        expected = domain_sizes[domain]
        row = [domain]

        for model in models:
            completed = 0
            total_agents = 0

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
                        stats = get_experiment_stats(file_path)
                        if stats and stats['total'] > 0:
                            break

                total_agents += 1
                if stats and stats['total'] == expected:
                    completed += 1

            if completed == 4:
                status = f'✅ {completed}/4 완료'
            elif completed > 0:
                status = f'⚠️ {completed}/4 부분완료'
            else:
                status = f'❌ {completed}/4 미완료'

            row.append(status)

        print(f'{row[0]:<15s} | {row[1]:<25s} | {row[2]:<25s} | {row[3]:<25s}')

    print('='*120)

if __name__ == '__main__':
    main()
