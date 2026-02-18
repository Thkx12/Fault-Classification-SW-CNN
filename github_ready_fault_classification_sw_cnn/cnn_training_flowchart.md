# Fluxograma - Treinamento da CNN (Espectrograma + Sliding Window)

```mermaid
flowchart TD
    A[Inicio] --> B[Carregar arquivos .mat<br/>Output/gaussian_noise_synthetic_batch/snr_20p0dB]
    B --> C[Inferir classe pelo nome do arquivo<br/>Normal BPFI BPFO Misalign Unbalance]
    C --> D[Split train val test por classe]
    D --> D1[Regra especial Normal<br/>split por janelas dentro do arquivo]
    D1 --> E[Selecionar canal e limpar NaN Inf]
    E --> F[Gerar janelas deslizantes<br/>window_size=2048 step=1024]
    F --> G[Converter cada janela em espectrograma<br/>Hann nperseg=256 noverlap=128 nfft=256 fmax=3000]
    G --> H[Montar dataset X y]
    H --> I[Normalizacao global com media e desvio do treino]
    I --> J[Balanceamento de classe<br/>WeightedRandomSampler + class weights]
    J --> K[Treinar CNN 2D<br/>Conv BN ReLU Pool + FC]
    K --> L[Validar por epoca]
    L --> M[Salvar melhor checkpoint por val_acc]
    M --> N[Teste final<br/>accuracy F1 matriz de confusao]
    N --> O[Salvar artefatos<br/>modelo relatorios curvas CSV]
    O --> P[Fim]
```

## Observacoes
- Script principal de treino: `Code/train_cnn_spectrogram_fault_classifier.py`
- Modelo gerado: `Output/cnn_spectrogram_fault_classifier_balanced_v2/spectrogram_cnn_model.pt`
- Inferencia sliding window: `Code/infer_cnn_spectrogram_sliding.py`
