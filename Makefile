.PHONY: install data train-all serve-model report test lint api docker-build docker-run

install:            ## install package + dev tools
	python -m pip install -r requirements-dev.txt && python -m pip install --no-deps -e .

data:               ## download + verify all datasets
	diabetes-xai download

train-all:          ## run every experiment (about 20-40 min on 2 CPU cores)
	diabetes-xai train configs/brfss_5050.yaml configs/brfss_xai.yaml configs/brfss_ensemble.yaml configs/pima_cv.yaml
	diabetes-xai report

serve-model:        ## train only the model served by the API (~1 min)
	diabetes-xai train configs/serve.yaml

report:
	diabetes-xai report

test:
	python -m pytest --cov

lint:
	ruff check src tests && ruff format --check src tests

api:
	MODEL_DIR=artifacts/brfss_serve uvicorn diabetes_xai.api.main:app --reload --port 8000

docker-build:
	docker build -t diabetes-xai:local .

docker-run:
	docker run --rm -p 8000:8000 diabetes-xai:local
