import streamlit as st
import plotly.graph_objects as go
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def criar_modelo():
    # 1. Definir a estrutura da rede bayesiana
    modelo = BayesianNetwork([
        ('Doença', 'Febre'),
        ('Doença', 'Tosse'),
        ('Doença', 'Espirros'),
        ('Estação', 'Espirros')
    ])

    # 2. Definir as Tabelas de Probabilidade Condicional (CPDs)
    cpd_doenca = TabularCPD(
        variable='Doença',
        variable_card=3,  # 0: Nenhuma, 1: Gripe, 2: Alergia
        values=[[0.7], [0.2], [0.1]]  # Probabilidades a priori
    )

    cpd_estacao = TabularCPD(
        variable='Estação',
        variable_card=4,  # 0: Inverno, 1: Primavera, 2: Verão, 3: Outono
        values=[[0.25], [0.25], [0.25], [0.25]]  # Todas as estações igualmente prováveis
    )

    cpd_febre = TabularCPD(
        variable='Febre',
        variable_card=2,  # 0: Não, 1: Sim
        values=[
            [0.95, 0.3, 0.8],   # P(Febre=0 | Doença)
            [0.05, 0.7, 0.2]    # P(Febre=1 | Doença)
        ],
        evidence=['Doença'],
        evidence_card=[3]
    )

    cpd_tosse = TabularCPD(
        variable='Tosse',
        variable_card=2,
        values=[
            [0.8, 0.2, 0.9],    # P(Tosse=0 | Doença)
            [0.2, 0.8, 0.1]     # P(Tosse=1 | Doença)
        ],
        evidence=['Doença'],
        evidence_card=[3]
    )

    cpd_espirros = TabularCPD(
        variable='Espirros',
        variable_card=2,
        values=[
            # P(Espirros=0 | Doença, Estação)
            [0.9, 0.6, 0.7, 0.8,  # Doença=0 (Nenhuma) para cada estação
             0.3, 0.1, 0.2, 0.2,  # Doença=1 (Gripe) para cada estação
             0.2, 0.05, 0.1, 0.1],  # Doença=2 (Alergia) para cada estação
            # P(Espirros=1 | Doença, Estação)
            [0.1, 0.4, 0.3, 0.2,  # Doença=0 (Nenhuma) para cada estação
             0.7, 0.9, 0.8, 0.8,  # Doença=1 (Gripe) para cada estação
             0.8, 0.95, 0.9, 0.9]  # Doença=2 (Alergia) para cada estação
            ],
        evidence=['Doença', 'Estação'],
        evidence_card=[3, 4]
    )

    try:
        # Adicionar CPDs ao modelo
        modelo.add_cpds(cpd_doenca, cpd_estacao, cpd_febre, cpd_tosse, cpd_espirros)
        
        # Verificar se o modelo é válido
        if not modelo.check_model():
            st.error("Erro: Modelo inconsistente!")
            return None, None
            
        # Criar inferência
        inferencia = VariableElimination(modelo)
        return modelo, inferencia
        
    except Exception as e:
        st.error(f"Erro ao criar modelo: {str(e)}")
        return None, None

def criar_interface():
    st.title("🏥 Sistema de Diagnóstico Médico - Rede Bayesiana")
    
    # Entradas do usuário
    with st.sidebar:
        st.header("🩺 Sintomas Observados")
        febre = st.selectbox("Febre:", ["Não", "Sim"])
        tosse = st.selectbox("Tosse:", ["Não", "Sim"])
        espirros = st.selectbox("Espirros:", ["Não", "Sim"])
        estacao = st.selectbox("Estação do Ano:", ["Inverno", "Primavera", "Verão", "Outono"])
    
    return febre, tosse, espirros, estacao

def exibir_resultados(resultado):
    st.subheader("📊 Probabilidades de Diagnóstico")
    doencas = ["Sem Doença", "Gripe", "Alergia"]
    
    # Métricas
    for i, prob in enumerate(resultado.values):
        st.metric(
            label=f"{doencas[i]}",
            value=f"{prob*100:.2f}%",
            delta="🔴 Risco Alto" if prob > 0.5 and i > 0 else "🟢 Risco Baixo"
        )
    
    # Gráfico de barras
    fig = go.Figure(data=[
        go.Bar(
            x=doencas,
            y=resultado.values * 100,
            marker_color=['green', 'red', 'orange']
        )
    ])
    
    fig.update_layout(
        title="Distribuição de Probabilidades",
        yaxis_title="Probabilidade (%)",
        xaxis_title="Diagnóstico",
        showlegend=False
    )
    
    st.plotly_chart(fig)

def main():
    # Criar modelo
    modelo, inferencia = criar_modelo()
    if modelo is None or inferencia is None:
        return
    
    # Criar interface
    febre, tosse, espirros, estacao = criar_interface()
    
    # Mapear entradas para valores numéricos
    mapeamento = {
        "Não": 0,
        "Sim": 1,
        "Inverno": 0, "Primavera": 1, "Verão": 2, "Outono": 3
    }
    
    try:
        # Calcular probabilidades
        resultado = inferencia.query(
            variables=['Doença'],
            evidence={
                'Febre': mapeamento[febre],
                'Tosse': mapeamento[tosse],
                'Espirros': mapeamento[espirros],
                'Estação': mapeamento[estacao]
            }
        )
        
        # Exibir resultados
        exibir_resultados(resultado)
        
        # Explicação
        with st.expander("🧠 Como a rede bayesiana funciona?"):
            st.markdown("""
            **Estrutura da Rede:**
            ```mermaid
            graph TD
                D[Doença] --> F[Febre]
                D --> T[Tosse]
                D --> E[Espirros]
                S[Estação] --> E
            ```
            
            **Variáveis:**
            - **Doença**: Causa raiz (Gripe, Alergia, Nenhuma)
            - **Estação**: Fator ambiental que influencia espirros
            - **Sintomas**: Febre, Tosse, Espirros (observáveis)
            
            **Inferência:**
            - Dados os sintomas observados, calculamos a probabilidade de cada doença usando o teorema de Bayes.
            - A rede considera relações complexas (ex: alergia é mais provável na primavera).
            """)
            
    except Exception as e:
        st.error(f"Erro ao realizar inferência: {str(e)}")
    
    st.caption("⚠️ Este é um modelo ilustrativo simplificado. Não use para diagnósticos reais!")

if __name__ == "__main__":
    main()