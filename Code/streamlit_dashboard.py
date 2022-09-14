import streamlit as st
import joblib
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

fname = '../Data/kobe_modelo.pkl'

############################################ SIDE BAR TITLE
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Previsor de Resultado Kobe, analisado e entrada de variáveis para avaliação de novas ou futuras jogadas
""")

st.sidebar.header('Tipo de Distância Analisada')
Field_Goal = st.sidebar.checkbox('Lance de 2 Pontos')
shot_type = '2PT Field Goal' if Field_Goal else '3PT Field Goal'
shot_type_por = 'Lance 2 Pontos' if Field_Goal else 'Lance 3 Pontos'

############################################ LEITURA DOS DADOS
@st.cache(allow_output_mutation=True)
def load_data(fname):
    return joblib.load(fname)

results = load_data(fname)
model = results[shot_type]['model'] 
train_data = results[shot_type]['data']
features = results[shot_type]['features']
target_col = results[shot_type]['target_col']
idx_train = train_data.categoria == 'treino'
idx_test = train_data.categoria == 'teste'
train_threshold = results[shot_type]['threshold']

print(f"features {features}")
print(f"train_data {train_data.columns}")


############################################ TITULO
st.title(f"""
Sistema Online de Avaliação de Lances Kobe - Tipo {shot_type_por}
""")

st.markdown(f"""
Esta interface pode ser utilizada para a explanação dos resultados
do modelo de classificação da qualidade dos lances à cesta do Kobe,
segundo as variáveis utilizadas para caracterizar as jogadas.

O modelo selecionado ({shot_type_por}) foi treinado com uma base total de {idx_train.sum()} e avaliado
com {idx_test.sum()} novos dados (histórico completo de {train_data.shape[0]} jogadas.

As jogadas são caracterizadas pelas seguintes variáveis: {features}.
""")


############################################ ENTRADA DE VARIAVEIS
st.sidebar.header('Entrada de Variáveis')
form = st.sidebar.form("input_form")
input_variables = {}

print(train_data.info())

for cname in features:
#     print(f'cname {cname}')
#     print(train_data[cname].unique())
#     print(train_data[cname].astype(float).max())
#     print(float(train_data[cname].astype(float).min()))
#     print(float(train_data[cname].astype(float).max()))
#     print(float(train_data[cname].astype(float).mean()))
    input_variables[cname] = (form.slider(cname.capitalize(),
                                          min_value = float(train_data[cname].astype(float).min()),
                                          max_value = float(train_data[cname].astype(float).max()),
                                          value = float(train_data[cname].astype(float).mean()))
                                   ) 
                             
form.form_submit_button("Avaliar")

############################################ PREVISAO DO MODELO 
@st.cache
def predict_user(input_variables):
    print(f'input_variables {input_variables}')
    X = pandas.DataFrame.from_dict(input_variables, orient='index').T
    Yhat = model.predict_proba(X)[0,1]
    return {
        'probabilidade': Yhat,
        'classificacao': int(Yhat >= train_threshold)
    }

user_kobe = predict_user(input_variables)

if user_kobe['classificacao'] == 0:
    st.sidebar.markdown("""Classificação:
    <span style="color:red">*Errou a cesta*</span>.
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""Classificação:
    <span style="color:green">*Acertou a cesta*</span>.
    """, unsafe_allow_html=True)

############################################ PAINEL COM AS PREVISOES HISTORICAS

fignum = plt.figure(figsize=(6,4))
for i in train_data.shot_made_flag.unique():
    sns.distplot(train_data[train_data[target_col] == i].probabilidade,
                 label = train_data[train_data[target_col] == i].target_label,
                 ax = plt.gca())
# User kobe
plt.plot(user_kobe['probabilidade'], 2, '*k', markersize=3, label='Lance Usuário')

plt.title('Resposta do Modelo para Jogadas Históricos')
plt.ylabel('Resultado Estimada')
plt.xlabel('Probabilidade de Acertar a Cesta')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')
st.pyplot(fignum)


