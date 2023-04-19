import streamlit as st

def page1():
    st.title('Page 1')
    st.write('This is the first page of my app.')
    
    # Create a link to page 2
    if st.button('Go to Page 2'):
        st.experimental_rerun()
        st.experimental_show('Page 2')
    
def page2():
    st.title('Page 2')
    st.write('This is the second page of my app.')
    
    # Create a link to page 1
    if st.button('Go to Page 1'):
        st.experimental_rerun()
        st.experimental_show('Page 1')

# Define a dictionary that maps the page names to the corresponding functions
pages = {
    'Page 1': page1,
    'Page 2': page2
}

list(pages.keys())
