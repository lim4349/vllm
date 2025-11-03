#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# OpenCUA-VL 관련 변경사항만 포함하여 vLLM 재빌드

set -e

echo "=========================================="
echo "OpenCUA-VL 전용 vLLM 재빌드"
echo "=========================================="

# 작업 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "1. 변경된 OpenCUA 관련 파일 확인..."
OPENCUA_FILES=(
    "vllm/model_executor/models/opencua_vl.py"
    "vllm/transformers_utils/configs/opencua_vl.py"
    "vllm/model_executor/models/vision.py"
    "vllm/model_executor/models/registry.py"
    "vllm/transformers_utils/configs/__init__.py"
)

for file in "${OPENCUA_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (없음)"
    fi
done

echo ""
echo "2. Python 캐시 삭제..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  ✓ 캐시 삭제 완료"

echo ""
echo "3. 기존 vLLM 설치 확인 및 제거..."
# uv 사용하는 경우
if command -v uv &> /dev/null; then
    echo "  uv를 사용하여 vLLM 제거 중..."
    uv pip uninstall vllm -y 2>/dev/null || echo "  vLLM이 설치되어 있지 않습니다"
else
    echo "  pip를 사용하여 vLLM 제거 중..."
    pip uninstall vllm -y 2>/dev/null || echo "  vLLM이 설치되어 있지 않습니다"
fi

echo ""
echo "4. 빌드 의존성 설치..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements/build.txt --torch-backend=auto || echo "  경고: 빌드 의존성 설치 중 일부 실패"
else
    pip install -r requirements/build.txt || echo "  경고: 빌드 의존성 설치 중 일부 실패"
fi

echo ""
echo "5. 일반 의존성 설치..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements/common.txt --torch-backend=auto || echo "  경고: 일반 의존성 설치 중 일부 실패"
else
    pip install -r requirements/common.txt || echo "  경고: 일반 의존성 설치 중 일부 실패"
fi

echo ""
echo "6. CUDA 의존성 설치 (xformers 제외)..."
if [ -f "requirements/cuda.txt" ]; then
    # xformers 라인만 제거 (include 라인은 유지)
    grep -v "^xformers" requirements/cuda.txt > /tmp/cuda_no_xformers.txt || true
    # 빈 파일 체크
    if [ -s /tmp/cuda_no_xformers.txt ]; then
        if command -v uv &> /dev/null; then
            uv pip install -r /tmp/cuda_no_xformers.txt --torch-backend=auto || echo "  경고: CUDA 의존성 설치 중 일부 실패"
        else
            pip install -r /tmp/cuda_no_xformers.txt || echo "  경고: CUDA 의존성 설치 중 일부 실패"
        fi
    else
        echo "  경고: 필터링된 cuda.txt 파일이 비어있습니다. 원본 파일 사용"
        if command -v uv &> /dev/null; then
            uv pip install -r requirements/cuda.txt --torch-backend=auto || echo "  경고: CUDA 의존성 설치 중 일부 실패"
        else
            pip install -r requirements/cuda.txt || echo "  경고: CUDA 의존성 설치 중 일부 실패"
        fi
    fi
    rm -f /tmp/cuda_no_xformers.txt
else
    echo "  경고: requirements/cuda.txt 파일이 없습니다"
fi

echo ""
echo "7. vLLM을 editable mode로 재설치 (OpenCUA 변경사항 포함)..."
echo "   이 과정은 10-30분 정도 걸릴 수 있습니다..."

if command -v uv &> /dev/null; then
    # xformers 스킵 설정
    export SKIP_XFORMERS_INSTALL=1
    uv pip install -e . --torch-backend=auto --no-build-isolation || {
        echo ""
        echo "✗ 설치 실패!"
        echo "수동으로 다시 시도하세요:"
        echo "  SKIP_XFORMERS_INSTALL=1 uv pip install -e . --torch-backend=auto --no-build-isolation"
        exit 1
    }
else
    pip install -e . --no-build-isolation || {
        echo ""
        echo "✗ 설치 실패!"
        echo "수동으로 다시 시도하세요:"
        echo "  pip install -e . --no-build-isolation"
        exit 1
    }
fi

echo ""
echo "=========================================="
echo "✓ OpenCUA-VL vLLM 재빌드 완료!"
echo "=========================================="
echo ""
echo "변경된 파일들:"
for file in "${OPENCUA_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done
echo ""
echo "설치 확인:"
echo "  python -c 'import vllm.model_executor.models.opencua_vl; print(\"OpenCUA-VL 모듈 로드 성공\")'"
echo ""
echo "서버 실행:"
echo "  vllm serve xlangai/OpenCUA-7B --tensor-parallel-size 1 ..."

