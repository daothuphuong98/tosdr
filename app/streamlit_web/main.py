# # py -m streamlit run C:\Users\Phuong\PycharmProjects\tosdr\app\streamlit_web\main.py
import streamlit as st
from app.utils import read_pdf, read_txt, read_docx
from app.service import TOSService
from app.utils import Methods

@st.cache(allow_output_mutation=True)
def load_service():
    return TOSService()

st.set_page_config(
    page_title="Terms of Service Detection",
    page_icon="📝",
    layout='wide'
)

service = load_service()
c1, c2 = st.columns([2.5,5])

with c2:
    st.title("Terms of Service Detection")

st.markdown('---')

bl, c1, bl, c2, bl = st.columns([0.1,1.5,0.1, 5,0.1])
with c1:
    upload_type = st.radio("Choose upload type", ['Text', 'Document'])
    stat = None
    raw_text = None
    ensemble = st.radio("Choose model type", ['Single model', 'Ensemble model'])
with c2:
    if upload_type == 'Text':
        MAX_WORDS = 1000
        raw_text = st.text_area(
            "Paste your text below (max %s words)" % MAX_WORDS,
            height=200,
        )

    elif upload_type == 'Document':
        doc = st.file_uploader("Document", type=['doc', 'docx', 'pdf', 'txt'])
        if doc:
            file_details = {"Filename": doc.name, "FileType": doc.type, "FileSize": doc.size}
            if file_details['FileType'] == "text/plain":
                stat, raw_text = read_txt(doc)
            elif file_details['FileType'] == 'application/pdf':
                stat, raw_text = read_pdf(doc)
            elif file_details['FileType'] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                stat, raw_text = read_docx(doc)

st.markdown('---')
bl, c1, bl, c2, bl = st.columns([0.1,1.5,0.1, 5,0.1])
with c1:
    with st.form(key="my_form"):
        if ensemble == 'Single model':
            ModelType = [st.selectbox("Choose your model", [Methods.BERT, Methods.RF, Methods.SVC, Methods.LGBM, Methods.ROBERTA])]
            threshold = st.text_input("Choose cutoff threshold")
            StopWordsCheckbox = st.checkbox(
                "Remove stop words",
                help="Tick this box to remove stop words from the document (currently English only)",
            )
            ensemble_method = 'Hard Voting'
        else:
            ModelType = st.multiselect("Choose your model", [Methods.BERT, Methods.RF, Methods.SVC, Methods.LGBM, Methods.ROBERTA])
            ensemble_method = st.radio("Choose ensemble method", ['Hard Voting', 'Soft Voting'])
            StopWordsCheckbox = st.checkbox(
                "Remove stop words",
                help="Tick this box to remove stop words from the document (currently English only)",
            )
            threshold=''

        submit_button = st.form_submit_button(label="Submit")

if submit_button:
    with c2:
        report = None
        if raw_text:
            if len(threshold) > 0:
                threshold = float(threshold.replace(',', '.'))
            else:
                threshold = None
            if ensemble_method == 'Hard Voting':
                voting = 'hard'
            else:
                voting = 'soft'
            detected = service.detect_paragraph(raw_text, ModelType, StopWordsCheckbox, threshold, voting=voting)
            report, roc_fig, conf_matrix_fig = service.score_test_dataset(ModelType, StopWordsCheckbox, voting=voting)
            st.markdown(detected['paragraph'], unsafe_allow_html=True)

    if report is not None:
        with st.expander("Click here to view the model's performance on test dataset"):
            bl, c1, bl, c2, bl, c3, bl = st.columns([0.1, 2.75, 0.1, 2, 0.1, 2, 0.1])
            with c1:
                st.subheader('Classification Report')
                st.text('\n\n⠀')
                st.text('⠀'+report[1:])
            with c2:
                st.subheader('ROC Curve')
                st.pyplot(roc_fig)
            with c3:
                st.subheader('Confusion Matrix')
                st.pyplot(conf_matrix_fig)
