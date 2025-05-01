import pandas as pd
import streamlit as st
from joblib import load  # To load the saved model

# Title and description
st.title("🔍 Homicide Prediction Tool")
st.write("This tool uses a trained model to predict the likelihood of homicide based on input data. Choose a model below.")

# Model selection
model_choice = st.selectbox("Select a Model", ["best_model_rf", "best_model_nn", "best_model_next"])

# Sliders for key features
physical_injury = st.slider("Physical Injury", 0.0, 1.0, 0.5)
robbery = st.slider("Robbery", 0.0, 1.0, 0.5)
theft = st.slider("Theft", 0.0, 1.0, 0.5)
homicide = st.slider("Homicide", 0.0, 1.0, 0.5)

# Load dataset to get column names
data = pd.read_csv("../dataset_pred2town_bel.csv", delimiter=";")
input_dict = {col: 0.0 for col in data.columns}

# Assign user inputs
input_dict["quant_lesao_corporal_crb"] = physical_injury
input_dict["quant_roubo_crb"] = robbery
input_dict["quant_furto_crb"] = theft
input_dict["quant_homicidio_crb"] = homicide

# Model selection & feature list
if model_choice == "best_model_rf":
    model = load("../best_model_rf.sav")
    model_features = model.feature_names_in_

elif model_choice == "best_model_nn":
    model = load("../best_model_nn.sav")
    model_features = [  # Exclude "Class"
        "mes_num_crb", "quant_lesao_corporal_crb", "quant_ameaca_crb", "quant_roubo_crb", 
        "quant_injuria_crb", "quant_furto_crb", "quant_lesao_no_transito_crb", 
        "quant_dano_no_transito_crb", "quant_difamacao_crb", "quant_homicidio_crb", 
        "quant_abandono_do_lar_crb", "quant_conflitos_vicinais_crb", "quant_conflitos_conjugais_crb", 
        "quant_fuga_do_lar_crb", "quant_estupro_de_vulneravel_crb", "quant_outros_fatos_atipicos_crb", 
        "quant_roubo_de_veiculo_crb", "quant_estelionato_crb", "quant_dano_crb", "quant_dano_civil_crb", 
        "quant_calunia_crb", "quant_conflitos_familiares_crb", "quant_trafico_de_drogas_crb", 
        "quant_vias_de_fato_crb", "quant_apropriacao_indebita_crb", "quant_agressao_fisica_crb", 
        "quant_receptacao_crb", "quant_estupro_crb", "quant_desaparecimento_de_pessoa_crb", 
        "quant_tentativa_de_homicidio_crb", "quant_poluicao_sonora_crb", "quant_outras_fraudes_crb", 
        "quant_desobediencia_crb", "quant_desacato_crb", "quant_perturbacoes_da_tranquilidade_crb"
    ]

elif model_choice == "best_model_next":
    model = load("../best_model_next.sav")
    model_features = [  # Same list as above,
        "mes_num_crb", "quant_lesao_corporal_crb", "quant_ameaca_crb", "quant_roubo_crb", 
        "quant_injuria_crb", "quant_furto_crb", "quant_lesao_no_transito_crb", 
        "quant_dano_no_transito_crb", "quant_difamacao_crb", "quant_homicidio_crb", 
        "quant_abandono_do_lar_crb", "quant_conflitos_vicinais_crb", "quant_conflitos_conjugais_crb", 
        "quant_fuga_do_lar_crb", "quant_estupro_de_vulneravel_crb", "quant_outros_fatos_atipicos_crb", 
        "quant_roubo_de_veiculo_crb", "quant_estelionato_crb", "quant_dano_crb", "quant_dano_civil_crb", 
        "quant_calunia_crb", "quant_conflitos_familiares_crb", "quant_trafico_de_drogas_crb", 
        "quant_vias_de_fato_crb", "quant_apropriacao_indebita_crb", "quant_agressao_fisica_crb", 
        "quant_receptacao_crb", "quant_estupro_crb", "quant_desaparecimento_de_pessoa_crb", 
        "quant_tentativa_de_homicidio_crb", "quant_poluicao_sonora_crb", "quant_outras_fraudes_crb", 
        "quant_desobediencia_crb", "quant_desacato_crb", "quant_perturbacoes_da_tranquilidade_crb"
    ]
    

# Fill in missing features with default because we are only testing with four columns
for feature in model_features:
    if feature not in input_dict:
        input_dict[feature] = 0.0

# Create input DataFrame and reorder columns
input_data = pd.DataFrame([input_dict])
input_data = input_data[model_features]

# Prediction
if st.button("Predict Homicide Risk"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ The model predicts a higher risk of homicide. It means this area is more likely to experience homicide.")
    else:
        st.warning("🚨 The model predicts a lower risk of homicide. It means this area is less likely to experience homicide.")
