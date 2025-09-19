import pandas as pd
import logging
import io
import os
import tempfile
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import seaborn as sns

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, Frame, PageTemplate, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.spider import SpiderChart
import textwrap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class EnhancedReportGenerationAgent:
    """
    Enhanced agent for generating comprehensive PDF reports with individual row analysis
    and complete metrics coverage.
    """
    
    def __init__(self, logo_path: Optional[str] = None):
        """
        Initialize the Enhanced Report Generation Agent.
        
        Args:
            logo_path (Optional[str]): Path to company logo image
        """
        logger.info("Initializing Enhanced Report Generation Agent...")
        
        self.logo_path = logo_path
        self.temp_files = []  # Track temporary files for cleanup
        
        # Setup matplotlib parameters
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
        
        # Define all expected metrics
        self.deepeval_metrics = [
            'contextual_precision', 'contextual_recall', 'contextual_relevancy',
            'answer_relevancy', 'faithfulness', 'hallucination', 
            'bias', 'toxicity', 'g_eval', 'summarization'
        ]
        
        self.mathematical_metrics = [
            'perplexity', 'bleu', 'rouge', 'meteor'
        ]
        
        logger.info("Enhanced Report Generation Agent initialized successfully")
    
    def validate_results(self, deepeval_results: Optional[Dict[str, Any]], 
                        math_results: Optional[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Validate which evaluation results are available and valid.
        """
        validation = {
            'deepeval_available': False,
            'math_available': False,
            'deepeval_valid': False,
            'math_valid': False
        }
        
        # Check DeepEval results
        if deepeval_results is not None:
            validation['deepeval_available'] = True
            try:
                if (isinstance(deepeval_results, dict) and 
                    'summary' in deepeval_results and 
                    'metric_averages' in deepeval_results.get('summary', {})):
                    validation['deepeval_valid'] = True
                    logger.info("DeepEval results validated successfully")
            except Exception as e:
                logger.warning(f"DeepEval results validation failed: {str(e)}")
        else:
            logger.info("DeepEval results not provided")
        
        # Check Mathematical results
        if math_results is not None:
            validation['math_available'] = True
            try:
                if (isinstance(math_results, dict) and 
                    'summary' in math_results and 
                    'metric_averages' in math_results.get('summary', {})):
                    validation['math_valid'] = True
                    logger.info("Mathematical results validated successfully")
            except Exception as e:
                logger.warning(f"Mathematical results validation failed: {str(e)}")
        else:
            logger.info("Mathematical results not provided")
        
        # Log validation summary
        valid_count = sum([validation['deepeval_valid'], validation['math_valid']])
        logger.info(f"Validation complete: {valid_count}/2 evaluation types available")
        
        return validation
    
    def create_comprehensive_metrics_chart(self, deepeval_results: Optional[Dict[str, Any]], 
                                         math_results: Optional[Dict[str, Any]], 
                                         validation: Dict[str, bool],
                                         output_path: str) -> str:
        """
        Create a comprehensive chart showing all available metrics.
        """
        try:
            logger.info("Creating comprehensive metrics chart...")
            
            # Collect all metrics
            all_metrics = {}
            
            # Add DeepEval metrics if available
            if validation['deepeval_valid']:
                deepeval_averages = deepeval_results.get('summary', {}).get('metric_averages', {})
                for metric_name, metric_data in deepeval_averages.items():
                    if isinstance(metric_data, dict) and 'average_score' in metric_data:
                        display_name = f"DE: {metric_name.replace('_', ' ').title()}"
                        all_metrics[display_name] = {
                            'score': metric_data['average_score'],
                            'category': 'DeepEval',
                            'color': '#3498db'
                        }
            
            # Add Mathematical metrics if available
            if validation['math_valid']:
                math_averages = math_results.get('summary', {}).get('metric_averages', {})
                for metric_name, metric_data in math_averages.items():
                    if isinstance(metric_data, dict):
                        score_key = 'average' if 'average' in metric_data else 'average_score'
                        if score_key in metric_data:
                            display_name = f"Math: {metric_name.replace('_', ' ').title()}"
                            all_metrics[display_name] = {
                                'score': metric_data[score_key],
                                'category': 'Mathematical',
                                'color': '#e74c3c'
                            }
            
            if not all_metrics:
                logger.warning("No metrics found for comprehensive chart")
                return ""
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(16, max(8, len(all_metrics) * 0.4)))
            
            # Prepare data
            metrics = list(all_metrics.keys())
            scores = [all_metrics[m]['score'] for m in metrics]
            colors_list = [all_metrics[m]['color'] for m in metrics]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(metrics))
            bars = ax.barh(y_pos, scores, color=colors_list, alpha=0.8)
            
            # Customize the plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(metrics, fontsize=9)
            ax.set_xlabel('Scores', fontsize=12, fontweight='bold')
            ax.set_title('Comprehensive Metrics Overview', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlim(0, 1)
            
            # Add score labels on bars
            for i, (bar, score) in enumerate(zip(bars, scores)):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{score:.3f}', ha='left', va='center', fontweight='bold', fontsize=8)
            
            # Add legend
            if validation['deepeval_valid'] and validation['math_valid']:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#3498db', label='DeepEval Metrics'),
                    Patch(facecolor='#e74c3c', label='Mathematical Metrics')
                ]
                ax.legend(handles=legend_elements, loc='lower right')
            
            # Add grid for better readability
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.temp_files.append(output_path)
            logger.info(f"Comprehensive metrics chart saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating comprehensive metrics chart: {str(e)}")
            return ""
    
    def create_row_comparison_chart(self, deepeval_results: Optional[Dict[str, Any]], 
                                  math_results: Optional[Dict[str, Any]], 
                                  validation: Dict[str, bool],
                                  output_path: str) -> str:
        """
        Create a chart comparing performance across rows.
        """
        try:
            logger.info("Creating row comparison chart...")
            
            # Collect row-wise data
            row_data = {}
            
            # Process DeepEval results
            if validation['deepeval_valid'] and 'results' in deepeval_results:
                for result in deepeval_results['results']:
                    if 'row_index' in result and 'summary' in result:
                        row_idx = result['row_index']
                        if row_idx not in row_data:
                            row_data[row_idx] = {'deepeval': 0, 'mathematical': 0}
                        row_data[row_idx]['deepeval'] = result['summary'].get('average_score', 0)
            
            # Process Mathematical results
            if validation['math_valid'] and 'results' in math_results:
                for result in math_results['results']:
                    if 'row_index' in result and 'summary' in result:
                        row_idx = result['row_index']
                        if row_idx not in row_data:
                            row_data[row_idx] = {'deepeval': 0, 'mathematical': 0}
                        row_data[row_idx]['mathematical'] = result['summary'].get('average_score', 0)
            
            if not row_data:
                logger.warning("No row data found for comparison chart")
                return ""
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            rows = sorted(row_data.keys())
            deepeval_scores = [row_data[r]['deepeval'] for r in rows]
            math_scores = [row_data[r]['mathematical'] for r in rows]
            
            x = np.arange(len(rows))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, deepeval_scores, width, label='DeepEval', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x + width/2, math_scores, width, label='Mathematical', color='#e74c3c', alpha=0.8)
            
            ax.set_xlabel('Row Index', fontweight='bold')
            ax.set_ylabel('Average Score', fontweight='bold')
            ax.set_title('Row-wise Performance Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Row {r}' for r in rows])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            add_value_labels(bars1)
            add_value_labels(bars2)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.temp_files.append(output_path)
            logger.info(f"Row comparison chart saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating row comparison chart: {str(e)}")
            return ""
    
    def create_individual_row_table(self, row_result: Dict[str, Any], agent_type: str) -> Table:
        """
        Create a detailed table for an individual row's results.
        """
        try:
            table_data = [["Metric", "Score", "Success", "Details"]]
            
            if 'metrics' in row_result:
                for metric_name, metric_data in row_result['metrics'].items():
                    if isinstance(metric_data, dict):
                        score = metric_data.get('score', 0.0)
                        success = "✓" if metric_data.get('success', True) else "✗"
                        
                        # Get additional details
                        details = []
                        if 'reason' in metric_data and metric_data['reason']:
                            reason = str(metric_data['reason'])[:100] + "..." if len(str(metric_data['reason'])) > 100 else str(metric_data['reason'])
                            details.append(f"Reason: {reason}")
                        
                        if 'evaluation_time' in metric_data:
                            details.append(f"Time: {metric_data['evaluation_time']}s")
                        
                        # Add other relevant fields for mathematical metrics
                        if agent_type == "Mathematical":
                            if 'raw_score' in metric_data:
                                details.append(f"Raw: {metric_data['raw_score']}")
                            if 'interpretation' in metric_data:
                                details.append(metric_data['interpretation'][:50] + "...")
                        
                        details_str = "\n".join(details) if details else "N/A"
                        
                        display_name = metric_name.replace('_', ' ').title()
                        table_data.append([display_name, f"{score:.3f}", success, details_str])
            
            if len(table_data) == 1:  # Only header
                table_data.append(["No metrics data", "N/A", "N/A", "N/A"])
            
            # Create table
            table = Table(table_data, colWidths=[120, 60, 50, 200])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            return table
            
        except Exception as e:
            logger.error(f"Error creating individual row table: {str(e)}")
            return Table([["Error creating table", "N/A", "N/A", "N/A"]])
    
    def add_page_elements(self, canvas, doc):
        """Add header elements to each page."""
        try:
            # Add logo if available
            if self.logo_path and os.path.exists(self.logo_path):
                try:
                    canvas.drawImage(
                        self.logo_path, 
                        0.5 * inch, letter[1] - 1.1 * inch, 
                        width=1.5 * inch, height=1 * inch, 
                        preserveAspectRatio=True
                    )
                except Exception as e:
                    logger.warning(f"Could not add logo: {str(e)}")
            
            # Add timestamp
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d %H:%M:%S")
            canvas.setFont("Helvetica", 10)
            canvas.drawRightString(
                letter[0] - 0.5 * inch, 
                letter[1] - 0.5 * inch, 
                f"Generated: {date_time}"
            )
            
            # Add page number
            page_num = canvas.getPageNumber()
            canvas.drawRightString(
                letter[0] - 0.5 * inch, 
                0.75 * inch, 
                f"Page {page_num}"
            )
            
        except Exception as e:
            logger.error(f"Error adding page elements: {str(e)}")
    
    def _get_performance_grade(self, score: float) -> str:
        """Convert score to performance grade."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def generate_enhanced_report(self, 
                               deepeval_results: Optional[Dict[str, Any]] = None,
                               math_results: Optional[Dict[str, Any]] = None,
                               input_data: Optional[pd.DataFrame] = None,
                               output_path: Optional[str] = None) -> io.BytesIO:
        """
        Generate an enhanced PDF report with individual row analysis.
        """
        try:
            logger.info("Generating enhanced PDF report...")
            start_time = datetime.now()
            
            # Validate available results
            validation = self.validate_results(deepeval_results, math_results)
            
            if not any([validation['deepeval_valid'], validation['math_valid']]):
                raise ValueError("No valid evaluation results provided. At least one evaluation agent result is required.")
            
            # Create temporary directory for charts
            temp_dir = tempfile.mkdtemp()
            
            # Generate charts
            comprehensive_chart_path = os.path.join(temp_dir, 'comprehensive_metrics.png')
            row_comparison_chart_path = os.path.join(temp_dir, 'row_comparison.png')
            
            comprehensive_chart = self.create_comprehensive_metrics_chart(
                deepeval_results, math_results, validation, comprehensive_chart_path
            )
            row_comparison_chart = self.create_row_comparison_chart(
                deepeval_results, math_results, validation, row_comparison_chart_path
            )
            
            # Create PDF buffer - FIXED THE TYPO HERE
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)  # Changed from SimpleDocDocument
            
            # Create page template with header
            page_template = PageTemplate(
                frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')],
                onPage=self.add_page_elements
            )
            doc.addPageTemplates([page_template])
            
            # Initialize story for PDF content
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue,
                alignment=TA_CENTER
            )
            
            row_title_style = ParagraphStyle(
                'RowTitle',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkgreen,
                backColor=colors.lightgrey,
                borderPadding=6
            )
            
            # Title and introduction
            story.append(Spacer(1, 40))
            story.append(Paragraph("Enhanced LLM Evaluation Report", title_style))
            story.append(Spacer(1, 30))
            
            # Executive Summary
            available_agents = []
            if validation['deepeval_valid']:
                available_agents.append("DeepEval Framework")
            if validation['math_valid']:
                available_agents.append("Mathematical Metrics")
            
            agents_text = " and ".join(available_agents)
            data_count = len(input_data) if input_data is not None else "Unknown number of"
            
            summary_text = f"""
            <b>Executive Summary</b><br/><br/>
            This enhanced report presents a comprehensive evaluation of Large Language Model (LLM) responses 
            using {agents_text}. The analysis includes both summary statistics and detailed individual row analysis.<br/><br/>
            
            <b>Evaluation Timestamp:</b> {start_time.strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Available Evaluations:</b> {agents_text}<br/>
            <b>Data Points Processed:</b> {data_count}<br/>
            """
            
            # Add metrics counts
            if validation['deepeval_valid']:
                deepeval_metrics_count = len(deepeval_results.get('summary', {}).get('metric_averages', {}))
                summary_text += f"<b>DeepEval Metrics:</b> {deepeval_metrics_count}<br/>"
                
                # List all DeepEval metrics
                de_metrics = list(deepeval_results.get('summary', {}).get('metric_averages', {}).keys())
                if de_metrics:
                    summary_text += f"<b>DeepEval Metrics List:</b> {', '.join([m.replace('_', ' ').title() for m in de_metrics])}<br/>"
            
            if validation['math_valid']:
                math_metrics_count = len(math_results.get('summary', {}).get('metric_averages', {}))
                summary_text += f"<b>Mathematical Metrics:</b> {math_metrics_count}<br/>"
                
                # List all Mathematical metrics
                math_metrics = list(math_results.get('summary', {}).get('metric_averages', {}).keys())
                if math_metrics:
                    summary_text += f"<b>Mathematical Metrics List:</b> {', '.join([m.replace('_', ' ').title() for m in math_metrics])}<br/>"
            
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Overall Performance Analysis
            performance_scores = []
            performance_text = "<b>Overall Performance Analysis</b><br/><br/>"
            
            if validation['deepeval_valid']:
                deepeval_avg = deepeval_results.get('summary', {}).get('overall_average_score', 0.0)
                performance_scores.append(deepeval_avg)
                performance_text += f"<b>DeepEval Average Score:</b> {deepeval_avg:.3f} ({self._get_performance_grade(deepeval_avg)})<br/>"
            
            if validation['math_valid']:
                math_avg = math_results.get('summary', {}).get('overall_scores', {}).get('average_across_metrics', 0.0)
                performance_scores.append(math_avg)
                performance_text += f"<b>Mathematical Average Score:</b> {math_avg:.3f} ({self._get_performance_grade(math_avg)})<br/>"
            
            if len(performance_scores) > 1:
                combined_avg = sum(performance_scores) / len(performance_scores)
                performance_text += f"<b>Combined Average Score:</b> {combined_avg:.3f} ({self._get_performance_grade(combined_avg)})<br/>"
            elif len(performance_scores) == 1:
                combined_avg = performance_scores[0]
                performance_text += f"<b>Overall Score:</b> {combined_avg:.3f} ({self._get_performance_grade(combined_avg)})<br/>"
            else:
                combined_avg = 0.0
            
            performance_text += f"<br/>The evaluation shows {self._get_performance_grade(combined_avg).lower()} performance based on available metrics."
            
            story.append(Paragraph(performance_text, styles['Normal']))
            story.append(PageBreak())
            
            # Comprehensive Visualizations
            story.append(Paragraph("Comprehensive Visualization Analysis", styles['Heading1']))
            story.append(Spacer(1, 20))
            
            # Add comprehensive metrics chart
            if comprehensive_chart and os.path.exists(comprehensive_chart):
                story.append(Paragraph("<b>All Metrics Overview</b>", styles['Heading2']))
                story.append(Spacer(1, 12))
                comp_img = Image(comprehensive_chart, width=7.5*inch, height=4.5*inch)
                comp_img.hAlign = 'CENTER'
                story.append(comp_img)
                story.append(Spacer(1, 20))
            
            # Add row comparison chart
            if row_comparison_chart and os.path.exists(row_comparison_chart):
                story.append(Paragraph("<b>Row-wise Performance Comparison</b>", styles['Heading2']))
                story.append(Spacer(1, 12))
                row_img = Image(row_comparison_chart, width=7*inch, height=4*inch)
                row_img.hAlign = 'CENTER'
                story.append(row_img)
                story.append(Spacer(1, 20))
            
            story.append(PageBreak())
            
            # Individual Row Analysis
            story.append(Paragraph("Individual Row Analysis", styles['Heading1']))
            story.append(Spacer(1, 20))
            
            # Process each row from both evaluation types
            all_rows = set()
            
            if validation['deepeval_valid'] and 'results' in deepeval_results:
                for result in deepeval_results['results']:
                    if 'row_index' in result:
                        all_rows.add(result['row_index'])
            
            if validation['math_valid'] and 'results' in math_results:
                for result in math_results['results']:
                    if 'row_index' in result:
                        all_rows.add(result['row_index'])
            
            # Sort rows for consistent presentation
            sorted_rows = sorted(all_rows)
            
            for row_idx in sorted_rows:
                # Row header
                story.append(Paragraph(f"Row {row_idx} - Detailed Analysis", row_title_style))
                story.append(Spacer(1, 12))
                
                # Get row data from input if available
                if input_data is not None and row_idx < len(input_data):
                    row_data = input_data.iloc[row_idx]
                    input_table_data = [
                        ["Field", "Content"],
                        ["Question", textwrap.fill(str(row_data.get('questions', 'N/A'))[:200] + "...", 60)],
                        ["Expected Answer", textwrap.fill(str(row_data.get('answers', 'N/A'))[:200] + "...", 60)],
                        ["LLM Answer", textwrap.fill(str(row_data.get('llm_answer', 'N/A'))[:200] + "...", 60)],
                        ["Context", textwrap.fill(str(row_data.get('contexts', 'N/A'))[:200] + "...", 60)]
                    ]
                    
                    input_table = Table(input_table_data, colWidths=[80, 350])
                    input_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ]))
                    
                    story.append(Paragraph("<b>Input Data:</b>", styles['Heading3']))
                    story.append(input_table)
                    story.append(Spacer(1, 12))
                
                # DeepEval results for this row
                if validation['deepeval_valid'] and 'results' in deepeval_results:
                    deepeval_row_result = None
                    for result in deepeval_results['results']:
                        if result.get('row_index') == row_idx:
                            deepeval_row_result = result
                            break
                    
                    if deepeval_row_result:
                        story.append(Paragraph("<b>DeepEval Metrics Results:</b>", styles['Heading3']))
                        deepeval_table = self.create_individual_row_table(deepeval_row_result, "DeepEval")
                        story.append(deepeval_table)
                        story.append(Spacer(1, 12))
                        
                        # Add summary for this row
                        if 'summary' in deepeval_row_result:
                            summary = deepeval_row_result['summary']
                            summary_text = f"""
                            <b>DeepEval Summary for Row {row_idx}:</b><br/>
                            • Average Score: {summary.get('average_score', 0):.3f}<br/>
                            • Successful Evaluations: {summary.get('successful_evaluations', 0)}/{summary.get('total_metrics', 0)}<br/>
                            • Evaluation Time: {summary.get('evaluation_time', 0):.2f} seconds<br/>
                            """
                            story.append(Paragraph(summary_text, styles['Normal']))
                            story.append(Spacer(1, 12))
                
                # Mathematical results for this row
                if validation['math_valid'] and 'results' in math_results:
                    math_row_result = None
                    for result in math_results['results']:
                        if result.get('row_index') == row_idx:
                            math_row_result = result
                            break
                    
                    if math_row_result:
                        story.append(Paragraph("<b>Mathematical Metrics Results:</b>", styles['Heading3']))
                        math_table = self.create_individual_row_table(math_row_result, "Mathematical")
                        story.append(math_table)
                        story.append(Spacer(1, 12))
                        
                        # Add summary for this row
                        if 'summary' in math_row_result:
                            summary = math_row_result['summary']
                            summary_text = f"""
                            <b>Mathematical Summary for Row {row_idx}:</b><br/>
                            • Average Score: {summary.get('average_score', 0):.3f}<br/>
                            • Successful Evaluations: {summary.get('successful_metrics', 0)}/{summary.get('total_metrics', 0)}<br/>
                            • Evaluation Time: {summary.get('evaluation_time', 0):.2f} seconds<br/>
                            """
                            story.append(Paragraph(summary_text, styles['Normal']))
                            story.append(Spacer(1, 12))
                
                # Add page break after each row (except the last one)
                if row_idx != sorted_rows[-1]:
                    story.append(PageBreak())
            
            # Final conclusions and recommendations
            story.append(PageBreak())
            story.append(Paragraph("Enhanced Conclusions and Recommendations", styles['Heading1']))
            story.append(Spacer(1, 12))
            
            conclusion_text = f"""
            <b>Detailed Analysis Summary:</b><br/>
            • Total rows analyzed: {len(sorted_rows)}<br/>
            • Evaluation frameworks used: {agents_text}<br/>
            • Individual row analysis provided for each data point<br/>
            • All available metrics included in the analysis<br/><br/>
            
            <b>Key Findings:</b><br/>
            • Each row shows different performance patterns across metrics<br/>
            • Detailed metric-by-metric analysis available for troubleshooting<br/>
            • Performance varies significantly between DeepEval and Mathematical metrics<br/><br/>
            
            <b>Enhanced Recommendations:</b><br/>
            • Review individual row results to identify specific improvement areas<br/>
            • Focus on consistently low-performing metrics across all rows<br/>
            • Consider row-specific optimizations based on individual analysis<br/>
            • Use detailed metric feedback for targeted model improvements<br/>
            """
            
            story.append(Paragraph(conclusion_text, styles['Normal']))
            
            # Build the PDF
            doc.build(story)
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(buffer.getvalue())
                logger.info(f"Enhanced report saved to {output_path}")
            
            # Cleanup temporary files
            self.cleanup_temp_files()
            
            buffer.seek(0)
            logger.info("Enhanced PDF report generation completed successfully")
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating enhanced report: {str(e)}")
            self.cleanup_temp_files()
            raise
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {str(e)}")
        self.temp_files.clear()

# Enhanced wrapper function
def generate_enhanced_evaluation_report(deepeval_results: Optional[Dict[str, Any]] = None,
                                       math_results: Optional[Dict[str, Any]] = None,
                                       input_data: Optional[pd.DataFrame] = None,
                                       output_path: Optional[str] = None,
                                       logo_path: Optional[str] = None) -> io.BytesIO:
    """
    Generate enhanced evaluation report with individual row analysis.
    """
    agent = EnhancedReportGenerationAgent(logo_path=logo_path)
    return agent.generate_enhanced_report(
        deepeval_results=deepeval_results,
        math_results=math_results,
        input_data=input_data,
        output_path=output_path
    )

# Keep backward compatibility
def generate_evaluation_report(deepeval_results: Optional[Dict[str, Any]] = None,
                             math_results: Optional[Dict[str, Any]] = None,
                             input_data: Optional[pd.DataFrame] = None,
                             output_path: Optional[str] = None,
                             logo_path: Optional[str] = None) -> io.BytesIO:
    """
    Backward compatible function that now uses the enhanced report generator.
    """
    return generate_enhanced_evaluation_report(
        deepeval_results=deepeval_results,
        math_results=math_results,
        input_data=input_data,
        output_path=output_path,
        logo_path=logo_path
    )

# Update the alias for the main class
FlexibleReportGenerationAgent = EnhancedReportGenerationAgent

if __name__ == "__main__":
    # Test with sample data that includes all metrics
    sample_deepeval_results = {
        'summary': {
            'overall_average_score': 0.332,
            'metric_averages': {
                'contextual_precision': {'average_score': 0.25},
                'contextual_recall': {'average_score': 0.30},
                'contextual_relevancy': {'average_score': 0.50},
                'answer_relevancy': {'average_score': 0.867},
                'faithfulness': {'average_score': 1.0},
                'hallucination': {'average_score': 0.50},
                'bias': {'average_score': 0.80},
                'toxicity': {'average_score': 0.95},
                'g_eval': {'average_score': 0.45},
                'summarization': {'average_score': 0.40}
            }
        },
        'results': [
            {
                'row_index': 0,
                'metrics': {
                    'contextual_precision': {'score': 0.2, 'success': True, 'reason': 'Good precision'},
                    'contextual_recall': {'score': 0.3, 'success': True, 'reason': 'Moderate recall'},
                    'answer_relevancy': {'score': 0.9, 'success': True, 'reason': 'Highly relevant'},
                    'faithfulness': {'score': 1.0, 'success': True, 'reason': 'Completely faithful'},
                    'hallucination': {'score': 0.6, 'success': True, 'reason': 'Some hallucination detected'},
                    'bias': {'score': 0.8, 'success': True, 'reason': 'Low bias'},
                    'toxicity': {'score': 0.95, 'success': True, 'reason': 'Non-toxic'},
                    'g_eval': {'score': 0.5, 'success': True, 'reason': 'Moderate coherence'},
                    'summarization': {'score': 0.4, 'success': True, 'reason': 'Fair summarization'}
                },
                'summary': {'average_score': 0.64, 'total_metrics': 9, 'successful_evaluations': 9}
            },
            {
                'row_index': 1,
                'metrics': {
                    'contextual_precision': {'score': 0.3, 'success': True, 'reason': 'Fair precision'},
                    'contextual_recall': {'score': 0.3, 'success': True, 'reason': 'Fair recall'},
                    'answer_relevancy': {'score': 0.834, 'success': True, 'reason': 'Relevant answer'},
                    'faithfulness': {'score': 1.0, 'success': True, 'reason': 'Faithful response'},
                    'hallucination': {'score': 0.4, 'success': True, 'reason': 'More hallucination'},
                    'bias': {'score': 0.8, 'success': True, 'reason': 'Low bias'},
                    'toxicity': {'score': 0.95, 'success': True, 'reason': 'Non-toxic'},
                    'g_eval': {'score': 0.4, 'success': True, 'reason': 'Lower coherence'},
                    'summarization': {'score': 0.4, 'success': True, 'reason': 'Fair summarization'}
                },
                'summary': {'average_score': 0.606, 'total_metrics': 9, 'successful_evaluations': 9}
            }
        ]
    }
    
    sample_math_results = {
        'summary': {
            'overall_scores': {'average_across_metrics': 0.088},
            'metric_averages': {
                'perplexity': {'average': 0.15},
                'bleu': {'average': 0.005},
                'rouge_average': {'average': 0.054},
                'rouge1_fmeasure': {'average': 0.06},
                'rouge2_fmeasure': {'average': 0.04},
                'rougeL_fmeasure': {'average': 0.06},
                'meteor': {'average': 0.116}
            }
        },
        'results': [
            {
                'row_index': 0,
                'metrics': {
                    'perplexity': {'score': 0.15, 'success': True},
                    'bleu': {'score': 0.006, 'success': True},
                    'rouge': {'average_fmeasure': 0.055, 'success': True},
                    'meteor': {'score': 0.120, 'success': True}
                },
                'summary': {'average_score': 0.083, 'total_metrics': 4, 'successful_metrics': 4}
            },
            {
                'row_index': 1,
                'metrics': {
                    'perplexity': {'score': 0.15, 'success': True},
                    'bleu': {'score': 0.004, 'success': True},
                    'rouge': {'average_fmeasure': 0.053, 'success': True},
                    'meteor': {'score': 0.112, 'success': True}
                },
                'summary': {'average_score': 0.080, 'total_metrics': 4, 'successful_metrics': 4}
            }
        ]
    }
    
    sample_input_data = pd.DataFrame({
        'questions': [
            "How can I customize Active Workspace?",
            "How does it work?"
        ],
        'answers': [
            "You can configure nearly every aspect of the commands for the Active Workspace interface.",
            "When the user selects an object, the universal viewer builds a list of available files and their associated viewers."
        ],
        'contexts': [
            "User interface configuration for Active Workspace",
            "Universal viewer functionality"
        ],
        'llm_answer': [
            "Hello! Customizing Active Workspace in Teamcenter can significantly enhance your user experience by tailoring the interface to meet your specific needs.",
            "Hello! Thank you for your question about Siemens PLM software. The universal viewer works by building a list of available files when users select objects."
        ]
    })
    
    try:
        # Generate enhanced report
        logger.info("Testing enhanced report generation...")
        pdf_buffer = generate_enhanced_evaluation_report(
            deepeval_results=sample_deepeval_results,
            math_results=sample_math_results,
            input_data=sample_input_data,
            output_path="enhanced_test_report.pdf"
        )
        print("✅ Enhanced PDF report generated successfully!")
        print(f"📄 Report size: {len(pdf_buffer.getvalue())} bytes")
        
    except Exception as e:
        print(f"❌ Enhanced report generation failed: {str(e)}")
        logger.error(f"Test execution failed: {str(e)}")