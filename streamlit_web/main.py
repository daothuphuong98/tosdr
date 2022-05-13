# # py -m streamlit run C:/Users/Phuong/PycharmProjects/tosdr/streamlit_web/main.py
import streamlit as st
from streamlit_web.utils import read_pdf, read_txt, read_docx
from streamlit_web.service import TOSService
from api.utils import Methods

@st.cache(allow_output_mutation=True)
def load_service():
    return TOSService()

st.set_page_config(
    page_title="Terms of Service Detection",
    page_icon="📝",
    layout='wide'
)

c1, c2 = st.columns([2.5,5])

with c2:
    st.title("Terms of Service Detection")

st.markdown('---')

bl, c1, bl, c2, bl = st.columns([0.1,1.5,0.1, 5,0.1])
with c1:
    upload_type = st.radio("Choose upload type", ['Text', 'Document'])
    stat = None
    raw_text = None
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
        ModelType = st.multiselect("Choose your model", ["BERT", Methods.RF, Methods.SVC, Methods.GB])
        ensemble_method = st.radio("Choose ensemble method", ['Hard Voting', 'Soft Voting'])

        # StopWordsCheckbox = st.checkbox(
        #     "Remove stop words",
        #     help="Tick this box to remove stop words from the document (currently English only)",
        # )

        submit_button = st.form_submit_button(label="Submit")

        if not submit_button:
            st.stop()
with c2:
    if raw_text:
        service = load_service()
        if ensemble_method == 'Hard Voting':
            detected = service.detect_paragraph(raw_text, ModelType)
        else:
            detected = service.detect_paragraph(raw_text, ModelType, voting='soft')
        st.markdown(detected['paragraph'], unsafe_allow_html=True)

