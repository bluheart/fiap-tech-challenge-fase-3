# fiap-tech-challenge-fase-3

# Medical Text Classifier API

## Decisão Arquitetural

### Estratégia de Deploy: AWS (Amazon Web Services)

**Justificativa da escolha AWS:**

1. **ECS Fargate para API Real-time:**
   - Serverless container management
   - Auto-scaling baseado em métricas
   - Integração nativa com CloudWatch e ALB
   - Baixa latência para inferência (< 100ms)

2. **S3 para Storage de Modelos:**
   - Versionamento de modelos
   - Alta durabilidade (99.999999999%)
   - Custo-efetivo para modelos pequenos

3. **MWAA (Managed Workflows for Apache Airflow):**
   - Orquestração gerenciada
   - Sem overhead de infraestrutura
   - Integração com serviços AWS

### Arquitetura Híbrida (Batch + Real-time)

**Real-time Inference (API):**
- API FastAPI em ECS Fargate
- Load Balancer para distribuição
- Auto-scaling baseado em CPU/RPS
- Latência alvo: < 100ms

**Batch Processing (Training):**
- Airflow DAG para pipeline de treino
- Execução diária/semanal
- Modelo treinado salvo no S3
- Atualização do modelo em produção

### Justificativa da Arquitetura

1. **Latência Crítica:** Classificação de urgência médica requer resposta rápida
2. **Volume Variável:** Hospital pode ter picos de demanda
3. **Custo-Efetividade:** Serverless quando possível, containers quando necessário
4. **Compliance:** AWS tem certificações HIPAA