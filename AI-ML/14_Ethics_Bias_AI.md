# 14. Ethics & Bias in AI

## 🎯 Learning Objectives
- Understand ethical implications of AI systems
- Learn to identify and mitigate various types of bias
- Master fairness metrics and algorithmic auditing
- Implement responsible AI practices
- Apply privacy-preserving techniques in ML

---

## 1. Understanding AI Ethics

**AI Ethics** encompasses the moral principles and values that guide the development and deployment of artificial intelligence systems.

### 1.1 Core Ethical Principles 🟢

#### Fundamental Principles:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EthicalAIFramework:
    """Framework for implementing ethical AI principles"""
    
    def __init__(self):
        self.principles = {
            'fairness': {
                'description': 'AI systems should treat all individuals and groups fairly',
                'metrics': ['demographic_parity', 'equalized_odds', 'calibration'],
                'implementation': 'bias_detection_and_mitigation'
            },
            'transparency': {
                'description': 'AI decision-making processes should be interpretable',
                'metrics': ['model_explainability', 'feature_importance', 'decision_paths'],
                'implementation': 'explainable_ai_techniques'
            },
            'accountability': {
                'description': 'Clear responsibility for AI system outcomes',
                'metrics': ['audit_trails', 'decision_logging', 'human_oversight'],
                'implementation': 'governance_frameworks'
            },
            'privacy': {
                'description': 'Protect individual privacy and data rights',
                'metrics': ['data_minimization', 'anonymization_quality', 'consent_tracking'],
                'implementation': 'privacy_preserving_techniques'
            },
            'beneficence': {
                'description': 'AI should benefit humanity and avoid harm',
                'metrics': ['impact_assessment', 'risk_evaluation', 'stakeholder_welfare'],
                'implementation': 'impact_assessment_frameworks'
            },
            'autonomy': {
                'description': 'Respect human agency and decision-making',
                'metrics': ['human_in_the_loop', 'opt_out_mechanisms', 'user_control'],
                'implementation': 'human_ai_collaboration'
            }
        }
    
    def assess_ethical_compliance(self, model, X_test, y_test, sensitive_attributes):
        """Assess model compliance with ethical principles"""
        
        assessment = {
            'timestamp': pd.Timestamp.now(),
            'principle_scores': {},
            'recommendations': [],
            'compliance_level': 'unknown'
        }
        
        # Fairness assessment
        fairness_score = self._assess_fairness(model, X_test, y_test, sensitive_attributes)
        assessment['principle_scores']['fairness'] = fairness_score
        
        # Transparency assessment
        transparency_score = self._assess_transparency(model)
        assessment['principle_scores']['transparency'] = transparency_score
        
        # Privacy assessment
        privacy_score = self._assess_privacy(X_test)
        assessment['principle_scores']['privacy'] = privacy_score
        
        # Overall compliance
        overall_score = np.mean(list(assessment['principle_scores'].values()))
        
        if overall_score >= 0.8:
            assessment['compliance_level'] = 'high'
        elif overall_score >= 0.6:
            assessment['compliance_level'] = 'medium'
        else:
            assessment['compliance_level'] = 'low'
            assessment['recommendations'].append('Immediate ethical review required')
        
        return assessment
    
    def _assess_fairness(self, model, X_test, y_test, sensitive_attributes):
        """Assess fairness across sensitive attributes"""
        if not sensitive_attributes:
            return 0.5  # Cannot assess without sensitive attributes
        
        predictions = model.predict(X_test)
        fairness_scores = []
        
        for attr in sensitive_attributes:
            if attr in X_test.columns:
                # Calculate demographic parity
                dp_score = self._demographic_parity(X_test[attr], predictions)
                fairness_scores.append(dp_score)
        
        return np.mean(fairness_scores) if fairness_scores else 0.5
    
    def _assess_transparency(self, model):
        """Assess model transparency and interpretability"""
        transparency_score = 0.0
        
        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            transparency_score += 0.3
        
        # Check if model is inherently interpretable
        interpretable_models = ['LogisticRegression', 'DecisionTreeClassifier', 'LinearRegression']
        if any(name in str(type(model)) for name in interpretable_models):
            transparency_score += 0.5
        
        # Check for predict_proba (uncertainty quantification)
        if hasattr(model, 'predict_proba'):
            transparency_score += 0.2
        
        return min(transparency_score, 1.0)
    
    def _assess_privacy(self, X_test):
        """Assess privacy protection measures"""
        privacy_score = 0.5  # Base score
        
        # Check for obvious PII columns (simple heuristic)
        pii_keywords = ['name', 'email', 'phone', 'ssn', 'id', 'address']
        pii_columns = [col for col in X_test.columns 
                      if any(keyword in col.lower() for keyword in pii_keywords)]
        
        if not pii_columns:
            privacy_score += 0.3  # No obvious PII found
        
        # Check for data range (normalized data suggests preprocessing)
        if all(X_test.select_dtypes(include=[np.number]).abs().max() <= 5):
            privacy_score += 0.2  # Data appears normalized
        
        return min(privacy_score, 1.0)
    
    def _demographic_parity(self, sensitive_attr, predictions):
        """Calculate demographic parity score"""
        groups = sensitive_attr.unique()
        group_rates = []
        
        for group in groups:
            group_mask = sensitive_attr == group
            if group_mask.sum() > 0:
                group_rate = predictions[group_mask].mean()
                group_rates.append(group_rate)
        
        if len(group_rates) < 2:
            return 1.0
        
        # Calculate fairness as 1 - max_difference
        max_diff = max(group_rates) - min(group_rates)
        return max(0, 1 - max_diff)
    
    def generate_ethics_report(self, assessment):
        """Generate comprehensive ethics report"""
        
        report = f"""
ETHICAL AI ASSESSMENT REPORT
Generated: {assessment['timestamp']}

OVERALL COMPLIANCE: {assessment['compliance_level'].upper()}

PRINCIPLE SCORES:
"""
        
        for principle, score in assessment['principle_scores'].items():
            status = "✓" if score >= 0.7 else "⚠" if score >= 0.5 else "✗"
            report += f"  {status} {principle.capitalize()}: {score:.2f}\n"
        
        if assessment['recommendations']:
            report += f"\nRECOMMENDATIONS:\n"
            for i, rec in enumerate(assessment['recommendations'], 1):
                report += f"  {i}. {rec}\n"
        
        report += f"\nDETAILED ANALYSIS:\n"
        for principle, details in self.principles.items():
            if principle in assessment['principle_scores']:
                score = assessment['principle_scores'][principle]
                report += f"\n{principle.upper()}:\n"
                report += f"  Score: {score:.2f}\n"
                report += f"  Description: {details['description']}\n"
                report += f"  Key Metrics: {', '.join(details['metrics'])}\n"
        
        return report

# Example usage
ethical_framework = EthicalAIFramework()

# Create synthetic dataset with sensitive attributes
np.random.seed(42)
n_samples = 1000

# Generate features
X = np.random.randn(n_samples, 5)
sensitive_attr = np.random.choice(['Group_A', 'Group_B'], n_samples, p=[0.7, 0.3])

# Create biased target (Group_B has lower positive rate)
base_prob = 1 / (1 + np.exp(-X[:, 0] - X[:, 1]))
bias_factor = np.where(sensitive_attr == 'Group_B', 0.7, 1.0)
y = np.random.binomial(1, base_prob * bias_factor)

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(5)]
df = pd.DataFrame(X, columns=feature_names)
df['sensitive_group'] = sensitive_attr
df['target'] = y

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df['target'], test_size=0.3, random_state=42
)

# Add sensitive attribute to test set for assessment
X_test_with_sensitive = X_test.copy()
X_test_with_sensitive['sensitive_group'] = df.loc[X_test.index, 'sensitive_group']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Assess ethical compliance
assessment = ethical_framework.assess_ethical_compliance(
    model, X_test_with_sensitive, y_test, ['sensitive_group']
)

# Generate ethics report
ethics_report = ethical_framework.generate_ethics_report(assessment)
print(ethics_report)
```

### 1.2 Stakeholder Impact Analysis 🟡

```python
class StakeholderImpactAnalysis:
    """Framework for analyzing AI system impact on different stakeholders"""
    
    def __init__(self):
        self.stakeholder_categories = {
            'primary_users': {
                'description': 'Direct users of the AI system',
                'impact_areas': ['user_experience', 'decision_outcomes', 'autonomy'],
                'assessment_methods': ['user_surveys', 'usage_analytics', 'outcome_tracking']
            },
            'affected_individuals': {
                'description': 'People affected by AI decisions but not direct users',
                'impact_areas': ['fairness', 'privacy', 'opportunity_access'],
                'assessment_methods': ['demographic_analysis', 'outcome_disparity', 'privacy_audit']
            },
            'organizations': {
                'description': 'Companies or institutions deploying the AI',
                'impact_areas': ['efficiency', 'risk', 'reputation', 'compliance'],
                'assessment_methods': ['performance_metrics', 'risk_assessment', 'audit_results']
            },
            'society': {
                'description': 'Broader societal impact',
                'impact_areas': ['employment', 'inequality', 'democratic_processes'],
                'assessment_methods': ['social_research', 'policy_analysis', 'longitudinal_studies']
            }
        }
    
    def conduct_impact_assessment(self, model, data, use_case_description):
        """Conduct comprehensive stakeholder impact assessment"""
        
        assessment = {
            'use_case': use_case_description,
            'timestamp': pd.Timestamp.now(),
            'stakeholder_impacts': {},
            'risk_levels': {},
            'mitigation_strategies': []
        }
        
        for stakeholder, details in self.stakeholder_categories.items():
            impact_score = self._assess_stakeholder_impact(
                stakeholder, model, data, use_case_description
            )
            
            assessment['stakeholder_impacts'][stakeholder] = {
                'impact_score': impact_score,
                'risk_level': self._categorize_risk(impact_score),
                'affected_areas': details['impact_areas'],
                'recommended_assessments': details['assessment_methods']
            }
        
        # Generate mitigation strategies
        assessment['mitigation_strategies'] = self._generate_mitigation_strategies(assessment)
        
        return assessment
    
    def _assess_stakeholder_impact(self, stakeholder, model, data, use_case):
        """Assess impact for specific stakeholder group"""
        
        # Simplified impact scoring based on model characteristics and use case
        base_impact = 0.5
        
        # High-risk use cases
        high_risk_keywords = ['criminal', 'hiring', 'lending', 'medical', 'legal', 'insurance']
        if any(keyword in use_case.lower() for keyword in high_risk_keywords):
            base_impact += 0.3
        
        # Model complexity (more complex = higher impact)
        if hasattr(model, 'n_estimators') and model.n_estimators > 100:
            base_impact += 0.1
        
        # Data sensitivity
        if data.shape[1] > 20:  # Many features might include sensitive data
            base_impact += 0.1
        
        # Stakeholder-specific adjustments
        if stakeholder == 'affected_individuals':
            base_impact += 0.2  # Higher impact on those without direct control
        elif stakeholder == 'society':
            base_impact += 0.1 if 'public' in use_case.lower() else 0
        
        return min(base_impact, 1.0)
    
    def _categorize_risk(self, impact_score):
        """Categorize risk level based on impact score"""
        if impact_score >= 0.8:
            return 'critical'
        elif impact_score >= 0.6:
            return 'high'
        elif impact_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_mitigation_strategies(self, assessment):
        """Generate mitigation strategies based on assessment"""
        
        strategies = []
        
        # Check for high-risk stakeholders
        high_risk_stakeholders = [
            name for name, details in assessment['stakeholder_impacts'].items()
            if details['risk_level'] in ['critical', 'high']
        ]
        
        if 'affected_individuals' in high_risk_stakeholders:
            strategies.append({
                'strategy': 'Enhanced fairness monitoring',
                'description': 'Implement continuous bias detection and fairness metrics',
                'priority': 'high'
            })
            strategies.append({
                'strategy': 'Transparent decision process',
                'description': 'Provide explanations for decisions affecting individuals',
                'priority': 'high'
            })
        
        if 'primary_users' in high_risk_stakeholders:
            strategies.append({
                'strategy': 'User control mechanisms',
                'description': 'Allow users to understand and influence AI decisions',
                'priority': 'medium'
            })
        
        if 'society' in high_risk_stakeholders:
            strategies.append({
                'strategy': 'Public engagement',
                'description': 'Involve public stakeholders in AI governance',
                'priority': 'medium'
            })
        
        return strategies

# Example stakeholder impact analysis
impact_analyzer = StakeholderImpactAnalysis()

use_case = "Automated hiring system for screening job applications"
impact_assessment = impact_analyzer.conduct_impact_assessment(
    model, X_test, use_case
)

print("STAKEHOLDER IMPACT ASSESSMENT")
print("=" * 50)
print(f"Use Case: {impact_assessment['use_case']}")
print(f"Assessment Date: {impact_assessment['timestamp']}")

for stakeholder, details in impact_assessment['stakeholder_impacts'].items():
    print(f"\n{stakeholder.upper().replace('_', ' ')}:")
    print(f"  Impact Score: {details['impact_score']:.2f}")
    print(f"  Risk Level: {details['risk_level'].upper()}")
    print(f"  Affected Areas: {', '.join(details['affected_areas'])}")

print("\nRECOMMENDED MITIGATION STRATEGIES:")
for i, strategy in enumerate(impact_assessment['mitigation_strategies'], 1):
    print(f"{i}. {strategy['strategy']} ({strategy['priority']} priority)")
    print(f"   {strategy['description']}")
```

---

## 2. Bias Detection and Mitigation

### 2.1 Types of Bias in ML 🟢

```python
class BiasDetector:
    """Comprehensive bias detection framework"""
    
    def __init__(self):
        self.bias_types = {
            'historical_bias': {
                'description': 'Bias present in training data due to past discrimination',
                'detection_methods': ['demographic_analysis', 'outcome_disparities'],
                'example': 'Hiring data showing historical gender bias in certain roles'
            },
            'representation_bias': {
                'description': 'Underrepresentation of certain groups in training data',
                'detection_methods': ['sample_size_analysis', 'demographic_distribution'],
                'example': 'Facial recognition trained mostly on one demographic group'
            },
            'measurement_bias': {
                'description': 'Systematic differences in data quality across groups',
                'detection_methods': ['data_quality_metrics', 'missing_data_analysis'],
                'example': 'Lower quality images for certain demographic groups'
            },
            'aggregation_bias': {
                'description': 'Assuming one model fits all subgroups equally well',
                'detection_methods': ['subgroup_performance_analysis', 'model_variance'],
                'example': 'Medical model trained on general population applied to specific demographics'
            },
            'evaluation_bias': {
                'description': 'Using inappropriate benchmarks for certain groups',
                'detection_methods': ['benchmark_analysis', 'metric_appropriateness'],
                'example': 'Language model evaluated only on formal text for diverse linguistic backgrounds'
            }
        }
    
    def detect_bias_comprehensive(self, X, y, sensitive_attributes, model=None):
        """Comprehensive bias detection across multiple dimensions"""
        
        bias_report = {
            'timestamp': pd.Timestamp.now(),
            'data_bias': {},
            'model_bias': {},
            'overall_bias_score': 0,
            'recommendations': []
        }
        
        # Data-level bias detection
        bias_report['data_bias'] = self._detect_data_bias(X, y, sensitive_attributes)
        
        # Model-level bias detection (if model provided)
        if model is not None:
            bias_report['model_bias'] = self._detect_model_bias(X, y, sensitive_attributes, model)
        
        # Calculate overall bias score
        data_score = np.mean([score for score in bias_report['data_bias'].values() if isinstance(score, (int, float))])
        model_score = np.mean([score for score in bias_report['model_bias'].values() if isinstance(score, (int, float))]) if bias_report['model_bias'] else 0
        
        bias_report['overall_bias_score'] = (data_score + model_score) / (2 if model else 1)
        
        # Generate recommendations
        bias_report['recommendations'] = self._generate_bias_recommendations(bias_report)
        
        return bias_report
    
    def _detect_data_bias(self, X, y, sensitive_attributes):
        """Detect bias in the dataset"""
        
        data_bias = {}
        
        for attr in sensitive_attributes:
            if attr not in X.columns:
                continue
            
            # Representation bias
            group_sizes = X[attr].value_counts()
            min_group_size = group_sizes.min()
            max_group_size = group_sizes.max()
            representation_ratio = min_group_size / max_group_size
            data_bias[f'{attr}_representation_bias'] = 1 - representation_ratio
            
            # Outcome bias (historical bias)
            if y is not None:
                group_outcomes = y.groupby(X[attr]).mean()
                outcome_variance = group_outcomes.var()
                data_bias[f'{attr}_outcome_bias'] = min(outcome_variance * 5, 1)  # Scale to 0-1
            
            # Data quality bias (missing values)
            missing_by_group = X.groupby(attr).apply(lambda x: x.isnull().sum().sum())
            if missing_by_group.sum() > 0:
                missing_variance = missing_by_group.var()
                total_missing = missing_by_group.sum()
                data_bias[f'{attr}_quality_bias'] = min(missing_variance / total_missing, 1)
        
        return data_bias
    
    def _detect_model_bias(self, X, y, sensitive_attributes, model):
        """Detect bias in model predictions"""
        
        model_bias = {}
        predictions = model.predict(X)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]  # Assuming binary classification
        else:
            probabilities = predictions.astype(float)
        
        for attr in sensitive_attributes:
            if attr not in X.columns:
                continue
            
            groups = X[attr].unique()
            
            # Demographic parity
            group_positive_rates = []
            for group in groups:
                group_mask = X[attr] == group
                if group_mask.sum() > 0:
                    positive_rate = predictions[group_mask].mean()
                    group_positive_rates.append(positive_rate)
            
            if len(group_positive_rates) > 1:
                demographic_parity_diff = max(group_positive_rates) - min(group_positive_rates)
                model_bias[f'{attr}_demographic_parity'] = demographic_parity_diff
            
            # Equalized odds
            if y is not None:
                equalized_odds_diffs = []
                for outcome in [0, 1]:
                    outcome_mask = y == outcome
                    if outcome_mask.sum() > 0:
                        group_rates = []
                        for group in groups:
                            group_mask = (X[attr] == group) & outcome_mask
                            if group_mask.sum() > 0:
                                rate = predictions[group_mask].mean()
                                group_rates.append(rate)
                        
                        if len(group_rates) > 1:
                            equalized_odds_diffs.append(max(group_rates) - min(group_rates))
                
                if equalized_odds_diffs:
                    model_bias[f'{attr}_equalized_odds'] = max(equalized_odds_diffs)
            
            # Calibration
            calibration_diffs = []
            for group in groups:
                group_mask = X[attr] == group
                if group_mask.sum() > 0 and y is not None:
                    group_probs = probabilities[group_mask]
                    group_outcomes = y[group_mask]
                    
                    # Simple calibration check: bin predictions and check calibration
                    bins = np.linspace(0, 1, 6)  # 5 bins
                    calibration_errors = []
                    
                    for i in range(len(bins) - 1):
                        bin_mask = (group_probs >= bins[i]) & (group_probs < bins[i + 1])
                        if bin_mask.sum() > 0:
                            predicted_prob = group_probs[bin_mask].mean()
                            actual_rate = group_outcomes[bin_mask].mean()
                            calibration_errors.append(abs(predicted_prob - actual_rate))
                    
                    if calibration_errors:
                        avg_calibration_error = np.mean(calibration_errors)
                        calibration_diffs.append(avg_calibration_error)
            
            if len(calibration_diffs) > 1:
                model_bias[f'{attr}_calibration'] = max(calibration_diffs) - min(calibration_diffs)
        
        return model_bias
    
    def _generate_bias_recommendations(self, bias_report):
        """Generate recommendations based on bias detection results"""
        
        recommendations = []
        
        # Check data bias
        for bias_type, score in bias_report['data_bias'].items():
            if isinstance(score, (int, float)) and score > 0.3:
                if 'representation' in bias_type:
                    recommendations.append({
                        'type': 'data_collection',
                        'message': f"Address representation bias in {bias_type.split('_')[0]} by collecting more diverse data",
                        'priority': 'high' if score > 0.6 else 'medium'
                    })
                elif 'outcome' in bias_type:
                    recommendations.append({
                        'type': 'data_preprocessing',
                        'message': f"Apply bias mitigation techniques to address historical bias in {bias_type.split('_')[0]}",
                        'priority': 'high' if score > 0.6 else 'medium'
                    })
                elif 'quality' in bias_type:
                    recommendations.append({
                        'type': 'data_quality',
                        'message': f"Improve data quality consistency across {bias_type.split('_')[0]} groups",
                        'priority': 'medium'
                    })
        
        # Check model bias
        for bias_type, score in bias_report['model_bias'].items():
            if isinstance(score, (int, float)) and score > 0.1:
                if 'demographic_parity' in bias_type:
                    recommendations.append({
                        'type': 'algorithmic_fairness',
                        'message': f"Apply demographic parity constraints for {bias_type.split('_')[0]}",
                        'priority': 'high' if score > 0.2 else 'medium'
                    })
                elif 'equalized_odds' in bias_type:
                    recommendations.append({
                        'type': 'algorithmic_fairness',
                        'message': f"Implement equalized odds constraints for {bias_type.split('_')[0]}",
                        'priority': 'high' if score > 0.2 else 'medium'
                    })
                elif 'calibration' in bias_type:
                    recommendations.append({
                        'type': 'model_calibration',
                        'message': f"Improve model calibration across {bias_type.split('_')[0]} groups",
                        'priority': 'medium'
                    })
        
        return recommendations

# Example bias detection
bias_detector = BiasDetector()

# Use the same biased dataset from earlier
bias_results = bias_detector.detect_bias_comprehensive(
    X_test_with_sensitive, y_test, ['sensitive_group'], model
)

print("COMPREHENSIVE BIAS DETECTION REPORT")
print("=" * 50)

print("\nDATA BIAS ANALYSIS:")
for bias_type, score in bias_results['data_bias'].items():
    if isinstance(score, (int, float)):
        severity = "HIGH" if score > 0.6 else "MEDIUM" if score > 0.3 else "LOW"
        print(f"  {bias_type}: {score:.3f} ({severity})")

print("\nMODEL BIAS ANALYSIS:")
for bias_type, score in bias_results['model_bias'].items():
    if isinstance(score, (int, float)):
        severity = "HIGH" if score > 0.2 else "MEDIUM" if score > 0.1 else "LOW"
        print(f"  {bias_type}: {score:.3f} ({severity})")

print(f"\nOVERALL BIAS SCORE: {bias_results['overall_bias_score']:.3f}")

print("\nRECOMMENDATIONS:")
for i, rec in enumerate(bias_results['recommendations'], 1):
    print(f"{i}. [{rec['priority'].upper()}] {rec['message']}")
```

### 2.2 Bias Mitigation Techniques 🟡

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels

class FairClassifier(BaseEstimator, ClassifierMixin):
    """Classifier with built-in fairness constraints"""
    
    def __init__(self, base_estimator=None, fairness_constraint='demographic_parity',
                 sensitive_features=None, lambda_fairness=1.0):
        self.base_estimator = base_estimator or LogisticRegression()
        self.fairness_constraint = fairness_constraint
        self.sensitive_features = sensitive_features or []
        self.lambda_fairness = lambda_fairness
    
    def fit(self, X, y, sensitive_X=None):
        """Fit the fair classifier"""
        
        # Check inputs
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        
        if sensitive_X is None:
            sensitive_X = X[:, self.sensitive_features] if self.sensitive_features else None
        
        # Train base model
        self.base_estimator.fit(X, y)
        
        # Apply post-processing fairness constraints
        if sensitive_X is not None:
            self._apply_fairness_constraints(X, y, sensitive_X)
        
        return self
    
    def _apply_fairness_constraints(self, X, y, sensitive_X):
        """Apply fairness constraints through post-processing"""
        
        predictions = self.base_estimator.predict_proba(X)[:, 1]
        
        if self.fairness_constraint == 'demographic_parity':
            self.threshold_map_ = self._demographic_parity_thresholds(
                predictions, sensitive_X
            )
        elif self.fairness_constraint == 'equalized_odds':
            self.threshold_map_ = self._equalized_odds_thresholds(
                predictions, y, sensitive_X
            )
        else:
            self.threshold_map_ = {group: 0.5 for group in np.unique(sensitive_X)}
    
    def _demographic_parity_thresholds(self, predictions, sensitive_X):
        """Calculate thresholds for demographic parity"""
        
        groups = np.unique(sensitive_X)
        target_rate = np.mean(predictions > 0.5)  # Overall positive rate
        
        threshold_map = {}
        
        for group in groups:
            group_mask = sensitive_X == group
            group_predictions = predictions[group_mask]
            
            # Find threshold that achieves target rate
            sorted_preds = np.sort(group_predictions)
            n_positive = int(len(sorted_preds) * target_rate)
            
            if n_positive < len(sorted_preds):
                threshold = sorted_preds[-(n_positive + 1)]
            else:
                threshold = sorted_preds[0] - 0.01
            
            threshold_map[group] = threshold
        
        return threshold_map
    
    def _equalized_odds_thresholds(self, predictions, y, sensitive_X):
        """Calculate thresholds for equalized odds"""
        
        groups = np.unique(sensitive_X)
        
        # Calculate target TPR and FPR (from overall population)
        overall_threshold = 0.5
        overall_preds = predictions > overall_threshold
        target_tpr = np.mean(overall_preds[y == 1])
        target_fpr = np.mean(overall_preds[y == 0])
        
        threshold_map = {}
        
        for group in groups:
            group_mask = sensitive_X == group
            group_predictions = predictions[group_mask]
            group_y = y[group_mask]
            
            # Find threshold that balances TPR and FPR
            best_threshold = 0.5
            best_score = float('inf')
            
            for threshold in np.linspace(0.1, 0.9, 50):
                group_preds = group_predictions > threshold
                
                if np.sum(group_y == 1) > 0:
                    tpr = np.mean(group_preds[group_y == 1])
                else:
                    tpr = 0
                
                if np.sum(group_y == 0) > 0:
                    fpr = np.mean(group_preds[group_y == 0])
                else:
                    fpr = 0
                
                score = abs(tpr - target_tpr) + abs(fpr - target_fpr)
                
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
            
            threshold_map[group] = best_threshold
        
        return threshold_map
    
    def predict(self, X, sensitive_X=None):
        """Make fair predictions"""
        
        X = check_array(X)
        
        if not hasattr(self, 'threshold_map_'):
            # No fairness constraints applied
            return self.base_estimator.predict(X)
        
        predictions = self.base_estimator.predict_proba(X)[:, 1]
        
        if sensitive_X is None:
            sensitive_X = X[:, self.sensitive_features] if self.sensitive_features else None
        
        if sensitive_X is None:
            # No sensitive features, use default threshold
            return (predictions > 0.5).astype(int)
        
        # Apply group-specific thresholds
        final_predictions = np.zeros(len(predictions))
        
        for group, threshold in self.threshold_map_.items():
            group_mask = sensitive_X == group
            final_predictions[group_mask] = (predictions[group_mask] > threshold).astype(int)
        
        return final_predictions
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        X = check_array(X)
        return self.base_estimator.predict_proba(X)

class BiasMetrics:
    """Comprehensive fairness metrics calculation"""
    
    @staticmethod
    def demographic_parity(y_pred, sensitive_attr):
        """Calculate demographic parity difference"""
        groups = np.unique(sensitive_attr)
        group_rates = []
        
        for group in groups:
            group_mask = sensitive_attr == group
            if group_mask.sum() > 0:
                rate = y_pred[group_mask].mean()
                group_rates.append(rate)
        
        return max(group_rates) - min(group_rates) if len(group_rates) > 1 else 0
    
    @staticmethod
    def equalized_odds(y_true, y_pred, sensitive_attr):
        """Calculate equalized odds difference"""
        groups = np.unique(sensitive_attr)
        max_tpr_diff = 0
        max_fpr_diff = 0
        
        tprs = []
        fprs = []
        
        for group in groups:
            group_mask = sensitive_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if len(group_y_true) > 0:
                # True Positive Rate
                if np.sum(group_y_true == 1) > 0:
                    tpr = np.mean(group_y_pred[group_y_true == 1])
                    tprs.append(tpr)
                
                # False Positive Rate
                if np.sum(group_y_true == 0) > 0:
                    fpr = np.mean(group_y_pred[group_y_true == 0])
                    fprs.append(fpr)
        
        max_tpr_diff = max(tprs) - min(tprs) if len(tprs) > 1 else 0
        max_fpr_diff = max(fprs) - min(fprs) if len(fprs) > 1 else 0
        
        return max(max_tpr_diff, max_fpr_diff)
    
    @staticmethod
    def calibration_difference(y_true, y_prob, sensitive_attr, n_bins=5):
        """Calculate calibration difference across groups"""
        groups = np.unique(sensitive_attr)
        calibration_errors = []
        
        for group in groups:
            group_mask = sensitive_attr == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            if len(group_y_true) > 0:
                # Calculate calibration error
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (group_y_prob > bin_lower) & (group_y_prob <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = group_y_true[in_bin].mean()
                        avg_confidence_in_bin = group_y_prob[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                calibration_errors.append(ece)
        
        return max(calibration_errors) - min(calibration_errors) if len(calibration_errors) > 1 else 0
    
    @staticmethod
    def comprehensive_fairness_report(y_true, y_pred, y_prob, sensitive_attr, sensitive_name='sensitive_group'):
        """Generate comprehensive fairness metrics report"""
        
        metrics = {
            'demographic_parity': BiasMetrics.demographic_parity(y_pred, sensitive_attr),
            'equalized_odds': BiasMetrics.equalized_odds(y_true, y_pred, sensitive_attr),
            'calibration_difference': BiasMetrics.calibration_difference(y_true, y_prob, sensitive_attr)
        }
        
        # Group-wise performance
        groups = np.unique(sensitive_attr)
        group_metrics = {}
        
        for group in groups:
            group_mask = sensitive_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if len(group_y_true) > 0:
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                
                group_metrics[f'{group}'] = {
                    'size': len(group_y_true),
                    'accuracy': accuracy_score(group_y_true, group_y_pred),
                    'precision': precision_score(group_y_true, group_y_pred, zero_division=0),
                    'recall': recall_score(group_y_true, group_y_pred, zero_division=0),
                    'positive_rate': group_y_pred.mean()
                }
        
        return {
            'fairness_metrics': metrics,
            'group_performance': group_metrics,
            'overall_performance': {
                'accuracy': accuracy_score(y_true, y_pred),
                'positive_rate': y_pred.mean()
            }
        }

# Example: Compare regular vs fair classifier
print("BIAS MITIGATION COMPARISON")
print("=" * 50)

# Regular classifier
regular_model = RandomForestClassifier(n_estimators=100, random_state=42)
regular_model.fit(X_train, y_train)
regular_pred = regular_model.predict(X_test)
regular_prob = regular_model.predict_proba(X_test)[:, 1]

# Fair classifier with demographic parity
fair_model_dp = FairClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    fairness_constraint='demographic_parity'
)

# Prepare sensitive features for training
sensitive_train = df.loc[X_train.index, 'sensitive_group'].values
sensitive_test = df.loc[X_test.index, 'sensitive_group'].values

fair_model_dp.fit(X_train, y_train, sensitive_train)
fair_pred_dp = fair_model_dp.predict(X_test, sensitive_test)
fair_prob_dp = fair_model_dp.predict_proba(X_test)[:, 1]

# Compare fairness metrics
print("\nREGULAR MODEL:")
regular_report = BiasMetrics.comprehensive_fairness_report(
    y_test, regular_pred, regular_prob, sensitive_test
)

for metric, value in regular_report['fairness_metrics'].items():
    print(f"  {metric}: {value:.3f}")

print("\nFAIR MODEL (Demographic Parity):")
fair_report = BiasMetrics.comprehensive_fairness_report(
    y_test, fair_pred_dp, fair_prob_dp, sensitive_test
)

for metric, value in fair_report['fairness_metrics'].items():
    print(f"  {metric}: {value:.3f}")

# Group performance comparison
print("\nGROUP PERFORMANCE COMPARISON:")
print("\nRegular Model:")
for group, metrics in regular_report['group_performance'].items():
    print(f"  {group}: Accuracy={metrics['accuracy']:.3f}, Positive Rate={metrics['positive_rate']:.3f}")

print("\nFair Model:")
for group, metrics in fair_report['group_performance'].items():
    print(f"  {group}: Accuracy={metrics['accuracy']:.3f}, Positive Rate={metrics['positive_rate']:.3f}")
```

---

## 3. Privacy-Preserving Machine Learning

### 3.1 Differential Privacy 🔴

```python
import hashlib
from typing import Tuple

class DifferentialPrivacy:
    """Implementation of differential privacy techniques"""
    
    def __init__(self, epsilon=1.0):
        """
        Initialize differential privacy with privacy budget epsilon
        
        Args:
            epsilon: Privacy budget (smaller = more private)
        """
        self.epsilon = epsilon
    
    def add_laplace_noise(self, value, sensitivity):
        """Add Laplace noise for differential privacy"""
        
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def add_gaussian_noise(self, value, sensitivity, delta=1e-5):
        """Add Gaussian noise for (epsilon, delta)-differential privacy"""
        
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    def private_mean(self, data, bounds=(0, 1)):
        """Calculate differentially private mean"""
        
        # Clamp data to bounds
        clamped_data = np.clip(data, bounds[0], bounds[1])
        
        # Sensitivity is the range divided by number of samples
        sensitivity = (bounds[1] - bounds[0]) / len(data)
        
        # Calculate noisy mean
        true_mean = np.mean(clamped_data)
        private_mean = self.add_laplace_noise(true_mean, sensitivity)
        
        return private_mean
    
    def private_count(self, data, condition_func):
        """Calculate differentially private count"""
        
        # Count how many elements satisfy condition
        true_count = np.sum([condition_func(x) for x in data])
        
        # Sensitivity for counting is 1
        sensitivity = 1
        
        # Add noise
        private_count = self.add_laplace_noise(true_count, sensitivity)
        
        return max(0, private_count)  # Count can't be negative
    
    def private_histogram(self, data, bins):
        """Create differentially private histogram"""
        
        # Create true histogram
        hist, bin_edges = np.histogram(data, bins=bins)
        
        # Add noise to each bin (sensitivity = 1 for histograms)
        sensitivity = 1
        private_hist = []
        
        for count in hist:
            noisy_count = self.add_laplace_noise(count, sensitivity)
            private_hist.append(max(0, noisy_count))  # Counts can't be negative
        
        return np.array(private_hist), bin_edges

class PrivateMLTraining:
    """Privacy-preserving machine learning training"""
    
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.dp = DifferentialPrivacy(epsilon)
    
    def private_gradient_descent(self, X, y, learning_rate=0.01, 
                               epochs=100, batch_size=32, clip_norm=1.0):
        """Differentially private stochastic gradient descent"""
        
        n_features = X.shape[1]
        weights = np.random.normal(0, 0.01, n_features)
        bias = 0
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process in batches
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Calculate gradients
                predictions = self._sigmoid(X_batch @ weights + bias)
                errors = predictions - y_batch
                
                # Clip gradients (for bounded sensitivity)
                grad_weights = X_batch.T @ errors / len(X_batch)
                grad_bias = np.mean(errors)
                
                # Clip gradient norm
                grad_norm = np.sqrt(np.sum(grad_weights**2) + grad_bias**2)
                if grad_norm > clip_norm:
                    grad_weights = grad_weights * clip_norm / grad_norm
                    grad_bias = grad_bias * clip_norm / grad_norm
                
                # Add noise for privacy
                sensitivity = 2 * clip_norm * learning_rate / batch_size
                
                noisy_grad_weights = grad_weights + np.random.normal(
                    0, sensitivity / self.epsilon, grad_weights.shape
                )
                noisy_grad_bias = grad_bias + np.random.normal(
                    0, sensitivity / self.epsilon
                )
                
                # Update weights
                weights -= learning_rate * noisy_grad_weights
                bias -= learning_rate * noisy_grad_bias
        
        return weights, bias
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow

class FederatedLearning:
    """Simulated federated learning framework"""
    
    def __init__(self, n_clients=5, privacy_enabled=True, epsilon=1.0):
        self.n_clients = n_clients
        self.privacy_enabled = privacy_enabled
        self.epsilon = epsilon
        self.global_model = None
        self.client_models = []
    
    def distribute_data(self, X, y, distribution='iid'):
        """Distribute data across clients"""
        
        n_samples = X.shape[0]
        
        if distribution == 'iid':
            # Randomly distribute data
            indices = np.random.permutation(n_samples)
            client_data = []
            
            samples_per_client = n_samples // self.n_clients
            
            for i in range(self.n_clients):
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client if i < self.n_clients - 1 else n_samples
                
                client_indices = indices[start_idx:end_idx]
                client_data.append((X[client_indices], y[client_indices]))
        
        elif distribution == 'non_iid':
            # Create non-IID distribution (group by label)
            client_data = [[] for _ in range(self.n_clients)]
            
            unique_labels = np.unique(y)
            for label in unique_labels:
                label_indices = np.where(y == label)[0]
                np.random.shuffle(label_indices)
                
                # Assign most data to one client, rest distributed
                primary_client = label % self.n_clients
                primary_size = int(0.7 * len(label_indices))
                
                client_data[primary_client].extend(label_indices[:primary_size])
                
                # Distribute remaining data
                remaining_indices = label_indices[primary_size:]
                for i, idx in enumerate(remaining_indices):
                    client_id = (primary_client + 1 + i) % self.n_clients
                    client_data[client_id].append(idx)
            
            # Convert to (X, y) tuples
            client_data = [(X[indices], y[indices]) for indices in client_data if indices]
        
        return client_data
    
    def federated_training(self, client_data, rounds=10, local_epochs=5):
        """Simulate federated learning training"""
        
        # Initialize global model
        n_features = client_data[0][0].shape[1]
        global_weights = np.random.normal(0, 0.01, n_features)
        global_bias = 0
        
        training_history = []
        
        for round_num in range(rounds):
            print(f"Federated Round {round_num + 1}/{rounds}")
            
            client_updates = []
            client_sizes = []
            
            # Client training
            for client_id, (X_client, y_client) in enumerate(client_data):
                if len(X_client) == 0:
                    continue
                
                # Start with global model
                local_weights = global_weights.copy()
                local_bias = global_bias
                
                # Local training
                if self.privacy_enabled:
                    private_trainer = PrivateMLTraining(self.epsilon)
                    local_weights, local_bias = private_trainer.private_gradient_descent(
                        X_client, y_client, epochs=local_epochs
                    )
                else:
                    # Standard local training
                    for epoch in range(local_epochs):
                        predictions = self._sigmoid(X_client @ local_weights + local_bias)
                        errors = predictions - y_client
                        
                        grad_weights = X_client.T @ errors / len(X_client)
                        grad_bias = np.mean(errors)
                        
                        local_weights -= 0.01 * grad_weights
                        local_bias -= 0.01 * grad_bias
                
                # Calculate update (difference from global model)
                weight_update = local_weights - global_weights
                bias_update = local_bias - global_bias
                
                client_updates.append((weight_update, bias_update))
                client_sizes.append(len(X_client))
            
            # Aggregate updates (FedAvg)
            if client_updates:
                total_samples = sum(client_sizes)
                
                # Weighted average of updates
                avg_weight_update = np.zeros_like(global_weights)
                avg_bias_update = 0
                
                for i, (weight_update, bias_update) in enumerate(client_updates):
                    weight = client_sizes[i] / total_samples
                    avg_weight_update += weight * weight_update
                    avg_bias_update += weight * bias_update
                
                # Update global model
                global_weights += avg_weight_update
                global_bias += avg_bias_update
            
            # Evaluate global model
            all_X = np.vstack([X for X, y in client_data if len(X) > 0])
            all_y = np.hstack([y for X, y in client_data if len(y) > 0])
            
            global_predictions = (self._sigmoid(all_X @ global_weights + global_bias) > 0.5).astype(int)
            accuracy = np.mean(global_predictions == all_y)
            
            training_history.append({
                'round': round_num + 1,
                'accuracy': accuracy,
                'n_participating_clients': len(client_updates)
            })
            
            print(f"  Global Model Accuracy: {accuracy:.3f}")
        
        self.global_model = (global_weights, global_bias)
        return training_history
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

# Example: Privacy-preserving techniques
print("PRIVACY-PRESERVING MACHINE LEARNING DEMO")
print("=" * 50)

# Differential Privacy example
print("\n1. DIFFERENTIAL PRIVACY EXAMPLE:")
np.random.seed(42)
sensitive_data = np.random.normal(50, 15, 1000)  # Simulated sensitive measurements

dp = DifferentialPrivacy(epsilon=0.5)  # Strong privacy

true_mean = np.mean(sensitive_data)
private_mean = dp.private_mean(sensitive_data, bounds=(0, 100))

print(f"True mean: {true_mean:.2f}")
print(f"Private mean (ε=0.5): {private_mean:.2f}")
print(f"Noise added: {abs(private_mean - true_mean):.2f}")

# Federated Learning example
print("\n2. FEDERATED LEARNING EXAMPLE:")

# Create federated dataset
X_fed, y_fed = make_classification(n_samples=1000, n_features=10, n_classes=2, 
                                  random_state=42, n_informative=8)

fed_learner = FederatedLearning(n_clients=5, privacy_enabled=True, epsilon=1.0)
client_data = fed_learner.distribute_data(X_fed, y_fed, distribution='non_iid')

print(f"Data distributed to {len(client_data)} clients")
for i, (X_client, y_client) in enumerate(client_data):
    if len(X_client) > 0:
        pos_rate = np.mean(y_client)
        print(f"  Client {i}: {len(X_client)} samples, {pos_rate:.2f} positive rate")

# Train federated model
history = fed_learner.federated_training(client_data, rounds=5, local_epochs=3)

print("\nFederated Training Progress:")
for record in history:
    print(f"  Round {record['round']}: Accuracy = {record['accuracy']:.3f}")
```

---

## 🎯 Key Takeaways

### Ethical AI Implementation:

#### Core Principles:
- **Fairness**: Ensure equitable treatment across all groups
- **Transparency**: Make AI decisions interpretable and explainable
- **Accountability**: Maintain clear responsibility chains
- **Privacy**: Protect individual data and rights
- **Beneficence**: Maximize benefits while minimizing harm
- **Autonomy**: Respect human agency and choice

#### Bias Mitigation Strategy:
- **Detection**: Use comprehensive metrics to identify bias
- **Prevention**: Address bias in data collection and model design
- **Correction**: Apply algorithmic fairness techniques
- **Monitoring**: Continuously assess fairness in production

#### Privacy Protection:
- **Differential Privacy**: Add noise to protect individual privacy
- **Federated Learning**: Train without centralizing data
- **Data Minimization**: Collect only necessary information
- **Anonymization**: Remove or obfuscate identifying information

#### Governance Framework:
- **Ethics Review Boards**: Establish oversight committees
- **Impact Assessments**: Evaluate potential societal effects
- **Audit Trails**: Maintain decision and model histories
- **Stakeholder Engagement**: Include affected communities in development

### Common Ethical Pitfalls:
1. **Bias amplification**: ML systems reinforcing existing inequalities
2. **Privacy violations**: Inadequate protection of personal data
3. **Lack of transparency**: "Black box" decisions affecting people's lives
4. **Insufficient oversight**: Deploying AI without proper governance
5. **Ignoring stakeholder impact**: Not considering all affected parties

---

## 📚 Next Steps

Complete your ML journey with:
- **[Tools & Frameworks](15_Tools_Frameworks.md)** - Complete development environment and tools
- **Review earlier topics** for deeper understanding and practice

---

*Next: [Tools & Frameworks →](15_Tools_Frameworks.md)*
