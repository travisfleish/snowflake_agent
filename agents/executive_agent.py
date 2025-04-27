"""
StrategicAdvisorAgent for transforming data analysis into executive recommendations.
Creates strategic insights and action plans from technical analysis results.
"""

import logging
import json
import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from utils.prompt_templates import AgentPromptTemplates

# Configure logger
logger = logging.getLogger(__name__)


class StrategicPriority(BaseModel):
    """Model for a strategic priority recommendation."""

    title: str = Field(..., description="Title of the strategic priority")
    description: str = Field(..., description="Detailed description of the priority")
    rationale: str = Field(..., description="Rationale based on data analysis")
    impact: Dict[str, Any] = Field(..., description="Expected business impact metrics")
    timeline: str = Field(..., description="Suggested implementation timeline")
    stakeholders: List[str] = Field(..., description="Key stakeholders to involve")
    resource_requirements: Dict[str, Any] = Field(..., description="Resources needed for implementation")
    success_metrics: List[str] = Field(..., description="Metrics to measure success")
    risks: List[Dict[str, str]] = Field(..., description="Potential risks and mitigation strategies")


class ExecutiveRecommendation(BaseModel):
    """Model for comprehensive executive recommendations."""

    id: str = Field(..., description="Unique identifier for the recommendation")
    title: str = Field(..., description="Title of the recommendation package")
    summary: str = Field(..., description="Executive summary (1-2 paragraphs)")
    context: str = Field(..., description="Business context and background")
    data_sources: List[str] = Field(..., description="Data sources used in analysis")
    key_findings: List[Dict[str, str]] = Field(..., description="Key findings from analysis")
    strategic_priorities: List[StrategicPriority] = Field(..., description="Strategic priorities and recommendations")
    next_steps: List[Dict[str, Any]] = Field(..., description="Suggested next steps and action plan")
    timestamp: str = Field(..., description="Timestamp of recommendation creation")


class StrategicAdvisorAgent(BaseAgent):
    """
    Agent that transforms technical analysis results into strategic recommendations.
    Specializes in executive communication, strategic planning, and business impact analysis.
    """

    def __init__(
            self,
            name: str = "Strategic Advisor",
            role: str = "Executive Strategy Consultant",
            goal: str = "Transform data insights into actionable strategic recommendations for executive leadership",
            backstory: str = None,
            business_context: Dict[str, Any] = None,
            industry_knowledge: Dict[str, Any] = None,
            recommendation_history: List[Dict[str, Any]] = None,
            **kwargs
    ):
        """
        Initialize a Strategic Advisor Agent.

        Args:
            name: Agent's name
            role: Agent's role description
            goal: Agent's main objective
            backstory: Agent's background story (optional)
            business_context: Context about the business (optional)
            industry_knowledge: Industry-specific knowledge (optional)
            recommendation_history: Previous recommendations (optional)
            **kwargs: Additional agent parameters
        """
        # Generate detailed backstory if not provided
        if backstory is None:
            backstory = self._generate_advisor_backstory()

        # Initialize base agent
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            backstory=backstory,
            **kwargs
        )

        # Store business context
        self.business_context = business_context or {
            "company_size": "Unknown",
            "industry": "General",
            "market_position": "Unknown",
            "strategic_goals": ["Growth", "Efficiency", "Innovation"],
            "key_challenges": ["Competition", "Market changes", "Internal alignment"],
            "risk_tolerance": "Moderate"
        }

        # Store industry knowledge
        self.industry_knowledge = industry_knowledge or {}

        # Initialize recommendation history
        self.recommendation_history = recommendation_history or []

        logger.info(f"Initialized {self.__class__.__name__}: {self.name}")

    def _generate_advisor_backstory(self) -> str:
        """
        Generate a detailed backstory for a strategic advisor.

        Returns:
            str: Detailed backstory
        """
        return (
            "I am a seasoned strategic advisor with over 20 years of experience helping executive teams "
            "translate complex data into actionable business strategies. My background spans management "
            "consulting at top-tier firms and C-suite advisory roles across multiple industries. "
            "I've guided organizations through digital transformations, market expansions, and competitive "
            "disruptions, always focusing on data-driven decision making and measurable business outcomes. "
            "My specialty is distilling technical insights into clear strategic priorities that align with "
            "business goals and organizational capabilities. I excel at communicating complex concepts in "
            "executive-friendly language, identifying high-impact opportunities, and developing pragmatic "
            "implementation roadmaps that drive meaningful results."
        )

    async def generate_strategic_recommendations(
            self,
            analysis_results: Dict[str, Any],
            business_question: str,
            executive_audience: str = "C-suite",
            strategic_focus: List[str] = None,
            time_horizon: str = "12 months",
            risk_threshold: str = "moderate"
    ) -> ExecutiveRecommendation:
        """
        Generate strategic recommendations from analysis results.

        Args:
            analysis_results: Data analysis results
            business_question: Original business question
            executive_audience: Target executive audience
            strategic_focus: Areas of strategic focus
            time_horizon: Time horizon for recommendations
            risk_threshold: Risk tolerance threshold

        Returns:
            ExecutiveRecommendation: Structured executive recommendations
        """
        # Set default strategic focus if not provided
        if strategic_focus is None:
            strategic_focus = ["revenue growth", "operational efficiency", "competitive positioning"]

        # Generate a unique ID for this recommendation
        rec_id = f"rec_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Step 1: Extract key findings from analysis results
        key_findings = await self._extract_key_findings(analysis_results, business_question)

        # Step 2: Generate strategic insights
        strategic_insights = await self._generate_strategic_insights(
            key_findings, business_question, strategic_focus
        )

        # Step 3: Develop strategic priorities
        strategic_priorities = await self._develop_strategic_priorities(
            strategic_insights,
            business_question,
            time_horizon,
            risk_threshold
        )

        # Step 4: Create implementation roadmap
        next_steps = await self._create_implementation_roadmap(
            strategic_priorities,
            executive_audience,
            time_horizon
        )

        # Step 5: Generate executive summary
        summary = await self._generate_executive_summary(
            key_findings,
            strategic_priorities,
            business_question
        )

        # Compile all components into a recommendation package
        recommendation = ExecutiveRecommendation(
            id=rec_id,
            title=f"Strategic Recommendations: {business_question}",
            summary=summary,
            context=await self._generate_business_context(business_question),
            data_sources=self._extract_data_sources(analysis_results),
            key_findings=key_findings,
            strategic_priorities=strategic_priorities,
            next_steps=next_steps,
            timestamp=datetime.datetime.now().isoformat()
        )

        # Store in recommendation history
        self.recommendation_history.append({
            "id": rec_id,
            "business_question": business_question,
            "timestamp": datetime.datetime.now().isoformat(),
            "recommendation": recommendation
        })

        return recommendation

    async def _extract_key_findings(
            self,
            analysis_results: Dict[str, Any],
            business_question: str
    ) -> List[Dict[str, str]]:
        """
        Extract key findings from analysis results.

        Args:
            analysis_results: Data analysis results
            business_question: Original business question

        Returns:
            List[Dict[str, str]]: Key findings with descriptions and implications
        """
        # Extract relevant information from analysis results
        results_summary = json.dumps(analysis_results, default=str)

        prompt = f"""
        You are an executive advisor extracting key findings from data analysis results.

        Business question: "{business_question}"

        Analysis results:
        {results_summary}

        Extract 5-7 key findings that are most relevant to the business question.
        For each finding:
        1. Provide a clear, concise title (1 line)
        2. Include a brief description of the finding (2-3 sentences)
        3. Explain the business implication (2-3 sentences)

        Each finding should be significant, data-backed, and directly relevant to executive decision-making.
        Focus on findings that have strategic importance, not just interesting data points.

        Return your response as a JSON array of objects, where each object has:
        - "title": The title of the finding
        - "description": Description of the finding
        - "implication": Business implication
        """

        # Call LLM to extract key findings
        self.thinking(f"Extracting key findings for question: {business_question}")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1500
        )

        # Parse findings from response
        try:
            findings_text = response['choices'][0]['message']['content']

            # Extract JSON from the response
            import re
            import json

            # Find JSON array in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', findings_text, re.DOTALL)
            if json_match:
                findings = json.loads(json_match.group(0))
            else:
                # Try to parse the entire response as JSON
                findings = json.loads(findings_text)

            return findings

        except Exception as e:
            logger.error(f"Error parsing key findings: {str(e)}")
            # Return a simplified structure if parsing fails
            return [
                {
                    "title": "Analysis Finding",
                    "description": "Finding extracted from analysis results.",
                    "implication": "Business implication of this finding."
                }
            ]

    async def _generate_strategic_insights(
            self,
            key_findings: List[Dict[str, str]],
            business_question: str,
            strategic_focus: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate strategic insights from key findings.

        Args:
            key_findings: Key findings from analysis
            business_question: Original business question
            strategic_focus: Areas of strategic focus

        Returns:
            List[Dict[str, Any]]: Strategic insights
        """
        # Format findings for LLM
        findings_text = "\n\n".join([
            f"Finding: {f.get('title')}\n"
            f"Description: {f.get('description')}\n"
            f"Implication: {f.get('implication')}"
            for f in key_findings
        ])

        # Format strategic focus areas
        focus_text = ", ".join(strategic_focus)

        prompt = f"""
        You are an executive strategic advisor translating data findings into strategic insights.

        Business question: "{business_question}"

        Strategic focus areas: {focus_text}

        Key findings from analysis:
        {findings_text}

        Based on these findings and focus areas, generate 3-5 strategic insights that could drive executive action.
        For each strategic insight:
        1. Provide a clear, actionable title
        2. Explain the insight and its strategic relevance
        3. Describe how it connects multiple findings
        4. Identify potential competitive advantage
        5. Note any critical dependencies or assumptions

        Each insight should be forward-looking, action-oriented, and directly tied to business value.
        Prioritize insights that align with the strategic focus areas.

        Return your response as a JSON array of objects with these fields.
        """

        # Call LLM to generate strategic insights
        self.thinking(f"Generating strategic insights for question: {business_question}")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1500
        )

        # Parse insights from response
        try:
            insights_text = response['choices'][0]['message']['content']

            # Extract JSON from the response
            import re
            import json

            # Find JSON array in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', insights_text, re.DOTALL)
            if json_match:
                insights = json.loads(json_match.group(0))
            else:
                # Try to parse the entire response as JSON
                insights = json.loads(insights_text)

            return insights

        except Exception as e:
            logger.error(f"Error parsing strategic insights: {str(e)}")
            # Return a simplified structure if parsing fails
            return [
                {
                    "title": "Strategic Insight",
                    "explanation": "Explanation of the strategic insight.",
                    "connection_to_findings": "How this connects to the findings.",
                    "competitive_advantage": "Potential competitive advantage.",
                    "dependencies": "Critical dependencies or assumptions."
                }
            ]

    async def _develop_strategic_priorities(
            self,
            strategic_insights: List[Dict[str, Any]],
            business_question: str,
            time_horizon: str,
            risk_threshold: str
    ) -> List[StrategicPriority]:
        """
        Develop strategic priorities from insights.

        Args:
            strategic_insights: Strategic insights
            business_question: Original business question
            time_horizon: Time horizon for recommendations
            risk_threshold: Risk tolerance threshold

        Returns:
            List[StrategicPriority]: Strategic priorities
        """
        # Format insights for LLM
        insights_text = "\n\n".join([
            f"Insight: {i.get('title')}\n"
            f"Explanation: {i.get('explanation')}\n"
            f"Competitive Advantage: {i.get('competitive_advantage')}\n"
            f"Dependencies: {i.get('dependencies')}"
            for i in strategic_insights
        ])

        # Include business context for more relevant recommendations
        context_text = f"""
        Business Context:
        - Industry: {self.business_context.get('industry')}
        - Company Size: {self.business_context.get('company_size')}
        - Market Position: {self.business_context.get('market_position')}
        - Strategic Goals: {', '.join(self.business_context.get('strategic_goals', []))}
        - Key Challenges: {', '.join(self.business_context.get('key_challenges', []))}
        """

        prompt = f"""
        You are a C-suite strategic advisor developing actionable strategic priorities.

        Business question: "{business_question}"

        Time horizon: {time_horizon}
        Risk tolerance: {risk_threshold}

        {context_text}

        Strategic insights:
        {insights_text}

        Develop 3 clear strategic priorities based on these insights. Each priority should:
        1. Be a specific, actionable recommendation
        2. Have clear business value and alignment with goals
        3. Be feasible within the time horizon
        4. Reflect the stated risk tolerance

        For each priority, provide:
        - Title: Clear, action-oriented title
        - Description: Detailed description of the strategic priority
        - Rationale: How this priority connects to data insights and business goals
        - Impact: Expected business impact with specific metrics
        - Timeline: Implementation timeline with key milestones
        - Stakeholders: Key stakeholders who need to be involved
        - Resource Requirements: People, technology, and budget needed
        - Success Metrics: How to measure success
        - Risks: Potential risks and mitigation strategies

        Return your response in JSON format matching the StrategicPriority model.
        """

        # Call LLM to develop strategic priorities
        self.thinking(f"Developing strategic priorities for question: {business_question}")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=2000
        )

        # Parse priorities from response
        try:
            priorities_text = response['choices'][0]['message']['content']

            # Extract JSON from the response
            import re
            import json

            # Find JSON array in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', priorities_text, re.DOTALL)
            if json_match:
                priorities_data = json.loads(json_match.group(0))
            else:
                # Try to parse the entire response as JSON
                priorities_data = json.loads(priorities_text)

            # Convert to StrategicPriority objects
            priorities = []
            for p_data in priorities_data:
                try:
                    priority = StrategicPriority(**p_data)
                    priorities.append(priority)
                except Exception as e:
                    logger.error(f"Error parsing priority: {str(e)}")

            return priorities

        except Exception as e:
            logger.error(f"Error parsing strategic priorities: {str(e)}")
            # Return a simplified structure if parsing fails
            return [
                StrategicPriority(
                    title="Strategic Priority",
                    description="Description of the strategic priority.",
                    rationale="Rationale for this priority.",
                    impact={"metric": "value"},
                    timeline="Implementation timeline.",
                    stakeholders=["Stakeholder 1", "Stakeholder 2"],
                    resource_requirements={"people": 0, "budget": 0},
                    success_metrics=["Metric 1", "Metric 2"],
                    risks=[{"risk": "Risk description", "mitigation": "Mitigation strategy"}]
                )
            ]

    async def _create_implementation_roadmap(
            self,
            strategic_priorities: List[StrategicPriority],
            executive_audience: str,
            time_horizon: str
    ) -> List[Dict[str, Any]]:
        """
        Create implementation roadmap for strategic priorities.

        Args:
            strategic_priorities: Strategic priorities
            executive_audience: Target executive audience
            time_horizon: Time horizon for implementation

        Returns:
            List[Dict[str, Any]]: Implementation roadmap
        """
        # Format priorities for LLM
        priorities_text = "\n\n".join([
            f"Priority: {p.title}\n"
            f"Description: {p.description}\n"
            f"Timeline: {p.timeline}\n"
            f"Stakeholders: {', '.join(p.stakeholders)}"
            for p in strategic_priorities
        ])

        prompt = f"""
        You are a strategic implementation advisor creating an action plan for executive priorities.

        Target audience: {executive_audience}
        Time horizon: {time_horizon}

        Strategic priorities:
        {priorities_text}

        Create a clear implementation roadmap with immediate next steps to execute these priorities.
        For each priority, provide:
        1. First 30 days: 2-3 specific actions to initiate implementation
        2. Key milestones: Major checkpoints across the implementation timeline
        3. Resource allocation: How to allocate resources across priorities
        4. Interdependencies: How these priorities interact with each other
        5. Quick wins: Early actions that can demonstrate value quickly

        The roadmap should be pragmatic, actionable, and appropriate for {executive_audience} to approve and sponsor.
        Focus on high-level direction rather than detailed project plans.

        Return your response as a JSON array of roadmap steps.
        """

        # Call LLM to create implementation roadmap
        self.thinking(f"Creating implementation roadmap for {len(strategic_priorities)} priorities")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1500
        )

        # Parse roadmap from response
        try:
            roadmap_text = response['choices'][0]['message']['content']

            # Extract JSON from the response
            import re
            import json

            # Find JSON array in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', roadmap_text, re.DOTALL)
            if json_match:
                roadmap = json.loads(json_match.group(0))
            else:
                # Try to parse the entire response as JSON
                roadmap = json.loads(roadmap_text)

            return roadmap

        except Exception as e:
            logger.error(f"Error parsing implementation roadmap: {str(e)}")
            # Return a simplified structure if parsing fails
            return [
                {
                    "priority": "Strategic Priority",
                    "first_30_days": ["Action 1", "Action 2"],
                    "key_milestones": ["Milestone 1", "Milestone 2"],
                    "resource_allocation": "Resource allocation strategy",
                    "interdependencies": "Dependencies with other priorities",
                    "quick_wins": ["Quick win 1", "Quick win 2"]
                }
            ]

    async def _generate_executive_summary(
            self,
            key_findings: List[Dict[str, str]],
            strategic_priorities: List[StrategicPriority],
            business_question: str
    ) -> str:
        """
        Generate an executive summary of findings and recommendations.

        Args:
            key_findings: Key findings from analysis
            strategic_priorities: Strategic priorities
            business_question: Original business question

        Returns:
            str: Executive summary
        """
        # Format findings and priorities for LLM
        findings_titles = "\n".join([f"- {f.get('title')}" for f in key_findings])
        priorities_titles = "\n".join([f"- {p.title}" for p in strategic_priorities])

        prompt = f"""
        You are a strategic advisor preparing an executive summary for the C-suite.

        Business question: "{business_question}"

        Key findings:
        {findings_titles}

        Strategic priorities:
        {priorities_titles}

        Write a compelling executive summary (1-2 paragraphs) that:
        1. Addresses the original business question
        2. Highlights the most significant findings
        3. Introduces the strategic priorities
        4. Conveys urgency and business impact

        Use confident, concise, executive-appropriate language. Focus on business outcomes, not methodology.
        The summary should be no more than 250 words and should motivate action.
        """

        # Call LLM to generate executive summary
        self.thinking(f"Generating executive summary for question: {business_question}")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=600
        )

        return response['choices'][0]['message']['content']

    async def _generate_business_context(
            self,
            business_question: str
    ) -> str:
        """
        Generate business context for recommendations.

        Args:
            business_question: Original business question

        Returns:
            str: Business context description
        """
        # Format business context for LLM
        context_text = f"""
        Business Context:
        - Industry: {self.business_context.get('industry')}
        - Company Size: {self.business_context.get('company_size')}
        - Market Position: {self.business_context.get('market_position')}
        - Strategic Goals: {', '.join(self.business_context.get('strategic_goals', []))}
        - Key Challenges: {', '.join(self.business_context.get('key_challenges', []))}
        """

        prompt = f"""
        You are a strategic advisor providing context for executive recommendations.

        Business question: "{business_question}"

        {context_text}

        Write a brief business context section (1 paragraph) that:
        1. Provides relevant background information
        2. Explains why this business question is important now
        3. Connects to broader strategic goals and market conditions

        The context should help executives understand why these recommendations matter
        and how they fit into the bigger picture of the organization's strategy.
        """

        # Call LLM to generate business context
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=400
        )

        return response['choices'][0]['message']['content']

    def _extract_data_sources(
            self,
            analysis_results: Dict[str, Any]
    ) -> List[str]:
        """
        Extract data sources from analysis results.

        Args:
            analysis_results: Data analysis results

        Returns:
            List[str]: Data sources used in analysis
        """
        # Extract data sources if available
        sources = []

        # Check for common source keys
        source_keys = ["data_sources", "sources", "tables", "datasets"]
        for key in source_keys:
            if key in analysis_results and isinstance(analysis_results[key], list):
                sources.extend(analysis_results[key])

        # Check profile for table names
        if "profile" in analysis_results and isinstance(analysis_results["profile"], dict):
            if "tables" in analysis_results["profile"]:
                sources.extend(analysis_results["profile"]["tables"])

        # Deduplicate sources
        sources = list(set(sources))

        # If no sources found, provide a generic source
        if not sources:
            sources = ["Analysis results"]

        return sources

    async def generate_executive_presentation(
            self,
            recommendation_id: str,
            format: str = "markdown",
            slides: int = 10
    ) -> str:
        """
        Generate an executive presentation from a recommendation.

        Args:
            recommendation_id: ID of the recommendation
            format: Presentation format (markdown, html, text)
            slides: Number of slides to generate

        Returns:
            str: Formatted presentation
        """
        # Find the recommendation in history
        recommendation = None
        for rec in self.recommendation_history:
            if rec["id"] == recommendation_id:
                recommendation = rec["recommendation"]
                break

        if not recommendation:
            return f"Error: Recommendation with ID {recommendation_id} not found"

        # Generate presentation outline
        outline = [
            "Title Slide",
            "Executive Summary",
            "Business Context",
            "Key Findings",
            "Strategic Priorities"
        ]

        # Add slides for each priority
        priority_titles = [p.title for p in recommendation.strategic_priorities]
        for title in priority_titles:
            outline.append(f"Priority: {title}")

        # Add implementation and next steps
        outline.append("Implementation Roadmap")
        outline.append("Next Steps")

        # Adjust to requested number of slides
        if len(outline) > slides:
            # Consolidate priorities if too many slides
            priority_index = outline.index("Strategic Priorities") + 1
            priorities_end = outline.index("Implementation Roadmap")
            outline[priority_index:priorities_end] = ["Strategic Priorities (Detail)"]

        # Generate presentation content
        presentation = await self._generate_presentation_content(
            recommendation, outline, format
        )

        return presentation

    async def _generate_presentation_content(
            self,
            recommendation: ExecutiveRecommendation,
            outline: List[str],
            format: str
    ) -> str:
        """
        Generate content for executive presentation.

        Args:
            recommendation: Recommendation data
            outline: Presentation outline
            format: Presentation format

        Returns:
            str: Formatted presentation content
        """
        # Format recommendation data for LLM
        rec_summary = {
            "title": recommendation.title,
            "summary": recommendation.summary,
            "context": recommendation.context,
            "key_findings": [f["title"] for f in recommendation.key_findings],
            "strategic_priorities": [p.title for p in recommendation.strategic_priorities],
            "priority_details": [
                {
                    "title": p.title,
                    "description": p.description,
                    "impact": p.impact,
                    "timeline": p.timeline
                }
                for p in recommendation.strategic_priorities
            ],
            "next_steps": recommendation.next_steps
        }

        rec_json = json.dumps(rec_summary, default=str)
        outline_text = "\n".join([f"- {slide}" for slide in outline])

        prompt = f"""
        You are creating an executive presentation based on strategic recommendations.

        Recommendation data:
        {rec_json}

        Presentation outline:
        {outline_text}

        Format: {format}

        Create a complete executive presentation following this outline. For each slide:
        1. Include a clear, impactful headline
        2. Add concise, focused content appropriate for executives
        3. Use bullet points for clarity and brevity
        4. Highlight key metrics and business impact

        The presentation should tell a cohesive story from business question through
        findings to recommendations and next steps.

        If format is markdown, use appropriate markdown formatting.
        If format is html, use appropriate html formatting.
        If format is text, use simple text formatting with clear slide separators.
        """

        # Call LLM to generate presentation
        self.thinking(f"Generating executive presentation for recommendation: {recommendation.id}")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=2500
        )

        return response['choices'][0]['message']['content']

    async def analyze_recommendation_impact(
            self,
            recommendation_id: str,
            business_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze potential impact of recommendations on business metrics.

        Args:
            recommendation_id: ID of the recommendation
            business_metrics: Business metrics to analyze (optional)

        Returns:
            Dict[str, Any]: Impact analysis
        """
        # Find the recommendation in history
        recommendation = None
        for rec in self.recommendation_history:
            if rec["id"] == recommendation_id:
                recommendation = rec["recommendation"]
                break

        if not recommendation:
            return {"error": f"Recommendation with ID {recommendation_id} not found"}

        # Set default metrics if not provided
        if business_metrics is None:
            business_metrics = ["revenue", "profitability", "market share", "customer satisfaction",
                                "operational efficiency"]

        # Format recommendation data for LLM
        priorities = [
            {
                "title": p.title,
                "description": p.description,
                "impact": p.impact,
                "timeline": p.timeline,
                "risks": p.risks
            }
            for p in recommendation.strategic_priorities
        ]

        priorities_json = json.dumps(priorities, default=str)
        metrics_text = ", ".join(business_metrics)

        prompt = f"""
        You are analyzing the potential business impact of strategic recommendations.

        Strategic priorities:
        {priorities_json}

        Business metrics to analyze: {metrics_text}

        For each business metric, analyze the potential impact of implementing all strategic priorities:
        1. Quantify the potential impact (estimate ranges where possible)
        2. Identify timeframes for realizing impact (short, medium, long-term)
        3. Describe the causal path from priorities to metric impact
        4. Note key dependencies and risks that could affect realization

        Also provide an overall impact assessment that weighs:
        - Cumulative potential across all metrics
        - Implementation feasibility
        - Time to realize value
        - Confidence level in the impact estimates

        Return your analysis as a JSON object with metrics as keys and impact assessments as values,
        plus an 'overall' key for the summary assessment.
        """

        # Call LLM to generate impact analysis
        self.thinking(f"Analyzing impact for recommendation: {recommendation_id}")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1500
        )

        # Parse impact analysis from response
        try:
            impact_text = response['choices'][0]['message']['content']

            # Extract JSON from the response
            import re
            import json

            # Find JSON object in the response
            json_match = re.search(r'\{.*\}', impact_text, re.DOTALL)
            if json_match:
                impact = json.loads(json_match.group(0))
            else:
                # Try to parse the entire response as JSON
                impact = json.loads(impact_text)

            return impact

        except Exception as e:
            logger.error(f"Error parsing impact analysis: {str(e)}")
            # Return a text-based response if parsing fails
            return {
                "impact_analysis": response['choices'][0]['message']['content']
            }

    async def compare_recommendations(
            self,
            recommendation_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple sets of recommendations.

        Args:
            recommendation_ids: List of recommendation IDs to compare

        Returns:
            Dict[str, Any]: Comparison analysis
        """
        # Find the recommendations in history
        recommendations = []
        for rec_id in recommendation_ids:
            for rec in self.recommendation_history:
                if rec["id"] == rec_id:
                    recommendations.append({
                        "id": rec_id,
                        "question": rec["business_question"],
                        "recommendation": rec["recommendation"]
                    })
                    break

        if not recommendations:
            return {"error": "No matching recommendations found"}

        # Format recommendations for LLM
        rec_summaries = []
        for rec in recommendations:
            priorities = [p.title for p in rec["recommendation"].strategic_priorities]
            rec_summaries.append({
                "id": rec["id"],
                "question": rec["question"],
                "summary": rec["recommendation"].summary,
                "priorities": priorities
            })

        rec_json = json.dumps(rec_summaries, default=str)

        prompt = f"""
        You are comparing multiple sets of strategic recommendations to identify patterns,
        conflicts, and synergies.

        Recommendations to compare:
        {rec_json}

        Provide a comprehensive comparison that addresses:
        1. Common themes and priorities across recommendations
        2. Key differences in strategic direction
        3. Potential conflicts in implementation or resource allocation
        4. Synergies that could be leveraged
        5. How these recommendations might be synthesized into a cohesive strategy

        Return your analysis as a structured JSON object with clear sections.
        """

        # Call LLM to generate comparison
        self.thinking(f"Comparing {len(recommendations)} sets of recommendations")
        response = await self._llm_client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1500
        )

        # Parse comparison from response
        try:
            comparison_text = response['choices'][0]['message']['content']

            # Extract JSON from the response
            import re
            import json

            # Find JSON object in the response
            json_match = re.search(r'\{.*\}', comparison_text, re.DOTALL)
            if json_match:
                comparison = json.loads(json_match.group(0))
            else:
                # Try to parse the entire response as JSON
                comparison = json.loads(comparison_text)

            return comparison

        except Exception as e:
            logger.error(f"Error parsing recommendation comparison: {str(e)}")
            # Return a text-based response if parsing fails
            return {
                "comparison": response['choices'][0]['message']['content']
            }

    def update_business_context(self, context_updates: Dict[str, Any]) -> None:
        """
        Update business context information.

        Args:
            context_updates: Updates to business context
        """
        self.business_context.update(context_updates)
        logger.info(f"Updated business context: {', '.join(context_updates.keys())}")


# Factory function to create a strategic advisor agent with default configuration
def create_strategic_advisor_agent(**kwargs) -> StrategicAdvisorAgent:
    """
    Create a Strategic Advisor Agent with default configuration.

    Args:
        **kwargs: Override default configuration parameters

    Returns:
        StrategicAdvisorAgent: Configured agent instance
    """
    return StrategicAdvisorAgent(**kwargs)