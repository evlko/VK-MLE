PROJECT="VK MLE Simple RecSys API"

run:
	uvicorn app.app:app --reload --host 0.0.0.0 --port 3000

tensorboard:
	tensorboard --logdir=model/tb_logs --bind_all

pretty:
	black .
	isort .

docker_build:
	docker image build -t vk-mle .

docker_run:
	docker run -p 3000:3000 vk-mle

.PHONY: run