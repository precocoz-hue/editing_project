# 로컬 Git으로 개인 작업·판단 근거 남기기

현재 `day6_compare` 폴더에서 실행합니다. GitHub 계정/원격 저장소는 필수가 아닙니다.
Git이 없다면 코드·보고서를 먼저 완료하고 강사에게 환경 문제를 알려 별도 기록을 남깁니다.

```bash
git --version
git init
git config user.name "본인 이름"
git config user.email "본인 이메일"
git add .gitignore README.md requirements-lock.txt main.py preprocessing.py evaluate.py checks.py search.py course_io.py
git add verify_outputs.py test_cases.py data reports Git_기록_안내.md
git commit -m "Set up Day6 model comparison"
```

위 이름·이메일은 자신의 것으로 바꾸며 설정은 이 저장소에만 적용됩니다.
코드를 완성하고 결과를 확인한 뒤 `reports/model_choice.md`에 선택 근거를 작성합니다.

```bash
git diff
git add main.py evaluate.py checks.py search.py reports/model_choice.md
git commit -m "Compare validation results and record limitations"
git status --short
git log --oneline -2
```

응용·심화에서는 첫 커밋 전에 `git add persist.py`로 저장·복원 모듈도 포함합니다. 이 실습의 `data/`는 제공된 합성 데이터이므로 저장소에 함께 넣습니다. 실제 업무 데이터는 이용 권한과 공개 범위를 먼저 확인해야 합니다.
`outputs/`와 모델 파일은 .gitignore로 제외됩니다. 제출용 압축에는 결과 CSV/JSON을 별도 포함합니다.
API 키·개인정보·가상환경·외부에서 받은 직렬화 모델을 커밋하지 않습니다.
파일을 수정한 것과 커밋된 것은 다릅니다. 실제 커밋과 diff를 확인하고 보고서에 hash를 기록합니다.
