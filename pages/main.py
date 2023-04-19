import streamlit as st

def page1():
    st.title('Page 1')
    st.write('This is the first page of my app.')
    
    # Create a clickable image that navigates to page 2
    st.write('Click on the image to go to Page 2')
    image = st.image('https://www.pixelstalk.net/wp-content/uploads/images6/Aesthetic-Cloud-Backgrounds-Free-Download.jpg', use_column_width=True)
    url = 'https://www.pixelstalk.net/wp-content/uploads/images6/Aesthetic-Cloud-Backgrounds-Free-Download.jpg'
    image.markdown(f'<a href="{url}">{image._repr_html_()}</a>', unsafe_allow_html=True)

def page2():
    st.title('Page 2')
    st.write('This is the second page of my app.')

# Define a dictionary that maps the page names to the corresponding functions
pages = {
    'Page 1': page1,
    'Page 2': page2
}

# Create a selectbox for page navigation
selection = st.sidebar.selectbox('Go to', list(pages.keys()))

# Call the appropriate page function based on the user's selection
pages[selection]()
