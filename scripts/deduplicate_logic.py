"""
GPT Logic 2,3,4-agent 중복 데이터 제거 스크립트
"""
import json
import sys
from pathlib import Path

# UTF-8 출력 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def deduplicate_results(input_file, output_file):
    """중복 제거"""
    print(f'\n처리 중: {input_file.name}')

    # 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'results' not in data:
        print('  ⚠️ results 필드 없음')
        return

    results = data['results']
    original_count = len(results)

    # question_id 기준 중복 제거 (첫 번째 결과만 유지)
    seen = set()
    unique_results = []

    for item in results:
        q_id = item.get('question_id')
        if q_id and q_id not in seen:
            seen.add(q_id)
            unique_results.append(item)

    # 정확도 재계산
    correct = sum(1 for item in unique_results if item.get('is_correct', False))
    total = len(unique_results)
    accuracy = (correct / total * 100) if total > 0 else 0

    # 데이터 업데이트
    data['results'] = unique_results
    data['total_questions'] = total
    data['correct'] = correct
    data['accuracy'] = accuracy

    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f'  원본: {original_count}개 → 중복 제거: {total}개')
    print(f'  정답: {correct}/{total} ({accuracy:.2f}%)')
    print(f'  ✅ 저장: {output_file}')

def main():
    merged_dir = Path('results/cross_domain/merged')

    print('='*80)
    print('GPT Logic 중복 데이터 제거')
    print('='*80)

    # Logic 2,3,4-agent 처리
    for agent in ['2', '3', '4']:
        input_file = merged_dir / f'logic_gpt_4o_{agent}agent.json'
        output_file = merged_dir / f'logic_gpt_4o_{agent}agent.json'

        if input_file.exists():
            deduplicate_results(input_file, output_file)
        else:
            print(f'\n⚠️ 파일 없음: {input_file.name}')

    print('\n' + '='*80)
    print('중복 제거 완료!')
    print('='*80)

if __name__ == '__main__':
    main()
