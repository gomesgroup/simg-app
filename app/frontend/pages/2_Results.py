import os
import time

import streamlit as st
from streamlit_molstar import st_molstar_content

import redis
from rq import Queue

import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
css = '''
<style>
    section.stMain > div {max-width:1200px}
    #stDecoration {background-color: #C41230; background-image: none;}
</style>
'''
st.markdown(css, unsafe_allow_html=True)


def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)


if 'job_id' not in st.query_params:
    st.switch_page("pages/1_Submit.py")

redis_url = os.environ.get('REDIS_HOST')
conn = redis.from_url(redis_url)

q = Queue(connection=conn)

job_id = st.query_params['job_id']
job = q.fetch_job(job_id)

if job.meta['status'] == 'Completed' and job.result:
    st_molstar_content(
        job.result['xyz'],
        'xyz'
    )

    cols = st.columns(2)
    cols[0].write(f"Results for {job.result['smiles']}")
    cols[0].image(
        job.result['graph'],
        caption="Generated graph (w/o 2nd order interactions)",
        use_column_width=True
    )

    cols[1].write("Interactions")
    cols[1].image(
        job.result['interactions'],
        caption="Interaction matrix",
        use_column_width=True
    )

    st.divider()

    cols = st.columns(2)
    cols[0].write("Explore specific interaction")
    n_atoms = job.result['is_atom'].sum()
    interaction_a = cols[0].selectbox(
        "Interaction A",
        np.arange(len(job.result['interaction_table'])) + n_atoms,
        index=None
    )
    interaction_b = cols[0].selectbox(
        "Interaction B",
        np.arange(len(job.result['interaction_table'])) + n_atoms,
        index=None
    )

    if interaction_a and interaction_b:
        probability = job.result['interaction_table'][interaction_a - n_atoms, interaction_b - n_atoms]
        cols[1].write(f"Probability: {probability:.3f}")
        cols[1].write(
            job.result['interaction_predictions'][
                (interaction_a - n_atoms) * len(job.result['interaction_table']) + (interaction_b - n_atoms)
                ]
        )

    st.divider()

    # Plot interaction table

    atom_targets = ['Charge', 'Core', 'Valence', 'Total']
    bond_targets = ['occupancy', 's', 'p', 'd', 'f', 'pol_diff', 'pol_coeff_diff']
    lone_pair_targets = ['s', 'p', 'd', 'f', 'occupancy']

    columns = [f"atom/{target}" for target in atom_targets] + \
              [f"lone_pair/{target}" for target in lone_pair_targets] + \
              [f"bond/{target}" for target in bond_targets]

    numbers = pd.DataFrame(
        job.result['node_level'],
        columns=columns
    ).map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

    numbers.loc[job.result['is_atom'], [f"lone_pair/{target}" for target in lone_pair_targets]] = ''
    numbers.loc[job.result['is_atom'], [f"bond/{target}" for target in bond_targets]] = ''
    numbers.loc[job.result['is_lp'], [f"atom/{target}" for target in atom_targets]] = ''
    numbers.loc[job.result['is_lp'], [f"bond/{target}" for target in bond_targets]] = ''
    numbers.loc[job.result['is_bond'], [f"atom/{target}" for target in atom_targets]] = ''
    numbers.loc[job.result['is_bond'], [f"lone_pair/{target}" for target in lone_pair_targets]] = ''

    st.table(
        numbers.drop(columns=['bond/d', 'bond/f', 'lone_pair/d', 'lone_pair/f'])
    )

else:
    st.write(f"Status: {job.meta['status']}")
    st.write("Refreshing the page in one second")
    st.write("Please wait...")
    time.sleep(1)
    st.rerun()
