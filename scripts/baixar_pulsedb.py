# ============================================================
# Script para baixar o PulseDB Supplementary Subsets via kagglehub
#
# Projeto: IC PIBIC PUC-SP - Estimativa de PA por PPG com IA
# Aluna: Leticia Barboza de Godoi | Orientador: Prof. Renan Trevisoli Doria
#
# Computador: notebook pessoal
# Destino: C:\IC_PPG\Datasets\PulseDB_Kaggle\
#
# Dataset: PulseDB Supplementary Subsets (Wang et al., 2023)
# Tamanho: ~19 GB | Tempo estimado: 10-60 min
# ============================================================

import os
import time
import kagglehub

print(f"kagglehub versao: {kagglehub.__version__}")
print(f"Cache configurado: {os.environ.get('KAGGLEHUB_CACHE', 'padrao do sistema')}")
print()

print("=" * 60)
print("Iniciando download do PulseDB Supplementary Subsets")
print("Tamanho: ~19 GB | Tempo estimado: 10-60 minutos")
print("=" * 60)
print()

tempo_inicio = time.time()

# Faz o download (oficial, sugerido pelo Kaggle)
# Se ja foi baixado, usa cache local (nao baixa de novo)
caminho_dataset = kagglehub.dataset_download(
    "weinanwangrutgers/pulsedb-balanced-training-and-testing"
)

tempo_total_min = (time.time() - tempo_inicio) / 60

print()
print("=" * 60)
print(f"Download concluido em {tempo_total_min:.1f} minutos.")
print(f"Caminho dos arquivos: {caminho_dataset}")
print("=" * 60)
print()

# Lista os arquivos baixados com tamanhos
print("Arquivos disponiveis:")
print("-" * 60)

tamanho_total_gb = 0.0

for nome_arquivo in sorted(os.listdir(caminho_dataset)):
    caminho_completo = os.path.join(caminho_dataset, nome_arquivo)
    if os.path.isfile(caminho_completo):
        tamanho_bytes = os.path.getsize(caminho_completo)
        tamanho_gb = tamanho_bytes / (1024 ** 3)
        tamanho_mb = tamanho_bytes / (1024 ** 2)
        tamanho_total_gb += tamanho_gb

        if tamanho_gb >= 1:
            print(f"  {nome_arquivo:<40} {tamanho_gb:>8.2f} GB")
        else:
            print(f"  {nome_arquivo:<40} {tamanho_mb:>8.2f} MB")

print("-" * 60)
print(f"  {'TOTAL':<40} {tamanho_total_gb:>8.2f} GB")
print()

print("Para usar nos seus notebooks, copie este caminho:")
print(f'  CAMINHO_PULSEDB = r"{caminho_dataset}"')
