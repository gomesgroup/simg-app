import streamlit as st


def app():
    st.set_page_config(layout="wide")
    css = '''
    <style>
        section.stMain > div {max-width:1000px}
        #stDecoration {background-color: #C41230; background-image: none;}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    cols = st.columns(3)
    cols[0].image('cmu.png')

    cols = st.columns([0.8, 0.2])
    cols[0].header("""Advancing Molecular Machine Learned Representations with Stereoelectronics-Infused Molecular Graphs""")
    cols[0].write("by Daniil A Boiko, Thiago Reschützegger, Benjamin Sanchez-Lengeling, Samuel M Blau, Gabe Gomes")
    cols[0].write("")

    if st.button("Submit"):
        st.write("Redirecting to the submit page")
        st.switch_page("pages/1_Submit.py")


app()
