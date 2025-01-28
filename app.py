import streamlit as st
from research_agent import ResearchAgent
from dotenv import load_dotenv

load_dotenv()

def initialize_session():
    if 'research_state' not in st.session_state:
        st.session_state.research_state = {
            'current_step': 0,
            'main_topic': '',
            'subtopics': [],
            'analyses': []
        }

def display_progress():
    # Updated to 3 steps only
    steps = ['Topic Input', 'Subtopic Generation', 'Research']
    current = st.session_state.research_state['current_step']
    st.progress((current + 1)/len(steps))
    st.subheader(f"Step {current+1}/{len(steps)}: {steps[current]}")

def main():
    st.set_page_config(page_title="AI Research Agent", layout="wide")
    initialize_session()
    
    st.title("DeepSeek Research Agent")
    display_progress()
    
    research_state = st.session_state.research_state
    
    # Step 1: Main Topic Input
    if research_state['current_step'] == 0:
        with st.form("main_topic_form"):
            main_topic = st.text_input("Enter research topic:")
            if st.form_submit_button("Start Research"):
                research_state['main_topic'] = main_topic
                research_state['current_step'] = 1
                st.rerun()
    
    # Step 2: Subtopic Generation
    elif research_state['current_step'] == 1:
        agent = ResearchAgent()
        with st.spinner("Generating subtopics..."):
            research_state['subtopics'] = agent.generate_subtopics(
                research_state['main_topic']
            )
        
        st.subheader("Generated Subtopics")
        for i, subtopic in enumerate(research_state['subtopics'], 1):
            st.write(f"{i}. {subtopic}")
        
        if st.button("Confirm Subtopics"):
            research_state['current_step'] = 2
            st.rerun()
    
    # Step 3: Research Execution (Final Step)
    elif research_state['current_step'] == 2:
        agent = ResearchAgent()
        analysis_container = st.container()
        
        # Only research if not already done
        if not research_state['analyses']:
            for i, subtopic in enumerate(research_state['subtopics']):
                with analysis_container.expander(f"Researching: {subtopic}", expanded=True):
                    placeholder = st.empty()
                    analysis = ""
                    
                    # Stream research results
                    for chunk in agent.client.chat(
                        prompt=f"Research subtopic: {subtopic}",
                        stream=True
                    ):
                        analysis += chunk
                        placeholder.markdown(analysis + "▌")
                    
                    # Store and display final analysis
                    placeholder.markdown(analysis)
                    research_state['analyses'].append({
                        "subtopic": subtopic,
                        "content": analysis
                    })
                    
                    # Add download button for each analysis
                    # st.download_button(
                    #     label=f"Download {subtopic} Analysis",
                    #     data=analysis,
                    #     file_name=f"{subtopic.replace(' ', '_')}_analysis.md",
                    #     mime="text/markdown",
                    #     key=f"download_{i}"
                    # )
        else:
            # Display existing analyses with download buttons
            for analysis in research_state['analyses']:
                with analysis_container.expander(f"Research: {analysis['subtopic']}", expanded=True):
                    st.markdown(analysis['content'])
                    st.download_button(
                        label=f"Download {analysis['subtopic']} Analysis",
                        data=analysis['content'],
                        file_name=f"{analysis['subtopic'].replace(' ', '_')}_analysis.md",
                        mime="text/markdown",
                        key=f"download_{analysis['subtopic']}"
                    )

if __name__ == "__main__":
    main()