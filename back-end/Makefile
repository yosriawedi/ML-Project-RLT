#prepare:
#	python3 main.py --prepare

#train:
#	python3 main.py --train

#evaluate:
#	python3 main.py --evaluate

#model: prepare train evaluate

#quality:
#	@echo "Formatage du code avec Black..."
#	black .
#	@echo "Vérification de la qualité du code avec Flake8..."
#	flake8 . --exclude venv
#	@echo "Analyse de sécurité avec Bandit..."
#	bandit -r . --exclude venv


#security:
#	@echo "Analyse de sécurité avec Bandit..."
#	@mkdir -p logs
#	@venv/bin/bandit -r . -f txt -o logs/security.log
#	@echo "Rapport généré : logs/security.log"

#save:
#	@echo "Sauvegarde du modèle..."
#	python3 -c "from main import main; main()"
# === Nouvelle étape pour tester l'API ===
#test-api:
#	@echo "Lancement de l'API FastAPI..."
#	uvicorn app:app --reload & sleep 2 && xdg-open http://127.0.0.1:8000/docs
# Variables
# Variables
IMAGE_NAME=nourrajhi/nour_rajhi_mlops
TAG=latest
PORT=8080

# -------------------------------------
# Préparer le projet
# -------------------------------------


preprocess:
	python3 main_RLT.py --preprocess

train2:
	python3 main_RLT.py --train2

predict:
	python3 main_RLT.py --predict

rlt: preprocess train2 predict

REQ_FILE = requirements.txt



# -------------------------------------
# Qualité et sécurité
# -------------------------------------
quality:
	@echo "Formatage du code avec Black..."
	black .
	@echo "Vérification de la qualité du code avec Flake8..."
	flake8 . --exclude venv
	@echo "Analyse de sécurité avec Bandit..."
	bandit -r . --exclude venv

security:
	@echo "Analyse de sécurité avec Bandit..."
	@mkdir -p logs
	@venv/bin/bandit -r . -f txt -o logs/security.log
	@echo "Rapport généré : logs/security.log"

save:
	@echo "Sauvegarde du modèle..."
	python3 -c "from main import main; main()"

test-api:
	@echo "Lancement de l'API FastAPI..."
	uvicorn app:app --reload & sleep 2 && xdg-open http://127.0.0.1:8000/docs

# -------------------------------------
# Docker
# -------------------------------------
build:
	docker build -t $(IMAGE_NAME):$(TAG) .

push:
	docker push $(IMAGE_NAME):$(TAG)

run:
	docker stop nour_rajhi_api || true
	docker rm nour_rajhi_api || true
	docker run -d -p $(PORT):8000 --name nour_rajhi_api $(IMAGE_NAME):$(TAG)

stop:
	docker stop nour_rajhi_api || true
	docker rm nour_rajhi_api || true

deploy: stop build push run


mlflow:
	mlflow ui --backend-store-uri "file:$$(pwd)/mlruns" --host 0.0.0.0 --port 5001



