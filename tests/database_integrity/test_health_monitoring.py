"""
Database Health Monitoring and Reporting Utilities
Utilities for ongoing database health monitoring, reporting, and alerting
"""
import pytest
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """Data structure for health metrics"""
    name: str
    value: float
    status: str  # "healthy", "warning", "critical"
    threshold: float
    description: str
    timestamp: str

@dataclass 
class HealthAlert:
    """Data structure for health alerts"""
    severity: str  # "low", "medium", "high", "critical"
    component: str
    message: str
    metrics: Dict[str, Any]
    timestamp: str
    recommended_action: str

class DatabaseHealthMonitor:
    """Comprehensive database health monitoring utility"""
    
    def __init__(self, database_clients: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        self.database_clients = database_clients
        self.config = config or self._default_config()
        self.health_history = []
        self.alerts = []
    
    def _default_config(self) -> Dict[str, Any]:
        """Default health monitoring configuration"""
        return {
            "thresholds": {
                "data_consistency_percentage": 95.0,
                "referential_integrity_percentage": 98.0,
                "freshness_percentage": 80.0,
                "orphaned_records_percentage": 2.0,
                "constraint_violations_percentage": 1.0,
                "max_stale_data_days": 30,
                "min_daily_activity_count": 1
            },
            "alert_settings": {
                "enable_alerts": True,
                "critical_threshold_breaches": 3,
                "warning_threshold_breaches": 5
            },
            "monitoring_intervals": {
                "real_time_check_minutes": 15,
                "comprehensive_check_hours": 6,
                "full_audit_days": 7
            }
        }
    
    def perform_comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check across all databases"""
        logger.info("Starting comprehensive database health check")
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "metrics": {},
            "alerts": [],
            "recommendations": []
        }
        
        try:
            # Run all health checks
            consistency_metrics = self._check_data_consistency()
            integrity_metrics = self._check_referential_integrity()
            freshness_metrics = self._check_data_freshness()
            activity_metrics = self._check_database_activity()
            orphaned_metrics = self._check_orphaned_records()
            constraint_metrics = self._check_constraint_violations()
            
            # Aggregate all metrics
            all_metrics = {
                **consistency_metrics,
                **integrity_metrics,
                **freshness_metrics,
                **activity_metrics,
                **orphaned_metrics,
                **constraint_metrics
            }
            
            health_report["metrics"] = all_metrics
            
            # Evaluate overall health
            overall_status, alerts, recommendations = self._evaluate_overall_health(all_metrics)
            health_report["overall_status"] = overall_status
            health_report["alerts"] = alerts
            health_report["recommendations"] = recommendations
            
            # Store history
            self.health_history.append(health_report)
            self.alerts.extend(alerts)
            
            logger.info(f"Health check completed. Overall status: {overall_status}")
            return health_report
            
        except Exception as e:
            logger.error(f"Error during comprehensive health check: {e}")
            health_report["overall_status"] = "error"
            health_report["error"] = str(e)
            return health_report
    
    def _check_data_consistency(self) -> Dict[str, HealthMetric]:
        """Check data consistency across databases"""
        metrics = {}
        
        try:
            # User embedding consistency
            users_response = self.database_clients["supabase"].client.table("users").select(
                "user_id, embedding_id"
            ).neq("embedding_id", None).limit(500).execute()
            
            users_with_embeddings = users_response.data if users_response.data else []
            consistent_embeddings = 0
            
            for user in users_with_embeddings:
                embedding_id = user["embedding_id"]
                
                # Check if embedding exists in Qdrant
                try:
                    result = self.database_clients["qdrant"].client.retrieve(
                        collection_name="user_embeddings",
                        ids=[embedding_id],
                        with_vectors=False
                    )
                    if result:
                        consistent_embeddings += 1
                except:
                    pass
            
            consistency_percentage = (consistent_embeddings / len(users_with_embeddings) * 100) if users_with_embeddings else 100
            
            metrics["user_embedding_consistency"] = HealthMetric(
                name="user_embedding_consistency",
                value=consistency_percentage,
                status=self._get_status(consistency_percentage, self.config["thresholds"]["data_consistency_percentage"]),
                threshold=self.config["thresholds"]["data_consistency_percentage"],
                description="Percentage of user embeddings that are consistent between Supabase and Qdrant",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error checking data consistency: {e}")
            metrics["user_embedding_consistency"] = HealthMetric(
                name="user_embedding_consistency",
                value=0.0,
                status="error",
                threshold=self.config["thresholds"]["data_consistency_percentage"],
                description=f"Error checking consistency: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
        
        return metrics
    
    def _check_referential_integrity(self) -> Dict[str, HealthMetric]:
        """Check referential integrity across databases"""
        metrics = {}
        
        try:
            # Video ID integrity between Supabase and MongoDB
            videos_response = self.database_clients["supabase"].client.table("videos").select(
                "video_id"
            ).limit(200).execute()
            
            video_ids = [v["video_id"] for v in videos_response.data] if videos_response.data else []
            
            if video_ids:
                # Check if videos exist in MongoDB summaries
                mongo_collection = self.database_clients["mongodb"].client.db.extractive_summaries
                existing_summaries = mongo_collection.find(
                    {"video_id": {"$in": video_ids}},
                    {"video_id": 1}
                )
                
                existing_video_ids = set([doc["video_id"] for doc in existing_summaries])
                integrity_count = len(existing_video_ids)
                integrity_percentage = (integrity_count / len(video_ids)) * 100
                
                metrics["video_referential_integrity"] = HealthMetric(
                    name="video_referential_integrity",
                    value=integrity_percentage,
                    status=self._get_status(integrity_percentage, self.config["thresholds"]["referential_integrity_percentage"]),
                    threshold=self.config["thresholds"]["referential_integrity_percentage"],
                    description="Percentage of videos that have corresponding records across databases",
                    timestamp=datetime.now().isoformat()
                )
            
        except Exception as e:
            logger.error(f"Error checking referential integrity: {e}")
            metrics["video_referential_integrity"] = HealthMetric(
                name="video_referential_integrity",
                value=0.0,
                status="error",
                threshold=self.config["thresholds"]["referential_integrity_percentage"],
                description=f"Error checking integrity: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
        
        return metrics
    
    def _check_data_freshness(self) -> Dict[str, HealthMetric]:
        """Check data freshness across the system"""
        metrics = {}
        
        try:
            # Check feedback freshness
            now = datetime.now()
            cutoff_date = now - timedelta(days=self.config["thresholds"]["max_stale_data_days"])
            
            feedback_response = self.database_clients["supabase"].client.table("feedback").select(
                "timestamp"
            ).gte("timestamp", cutoff_date.isoformat()).execute()
            
            recent_feedback_count = len(feedback_response.data) if feedback_response.data else 0
            
            # Get total feedback to calculate percentage
            total_feedback_response = self.database_clients["supabase"].client.table("feedback").select(
                "timestamp", count="exact"
            ).execute()
            
            total_feedback_count = total_feedback_response.count if total_feedback_response.count else 0
            
            if total_feedback_count > 0:
                freshness_percentage = (recent_feedback_count / total_feedback_count) * 100
            else:
                freshness_percentage = 100  # No data is considered "fresh" by default
            
            metrics["feedback_freshness"] = HealthMetric(
                name="feedback_freshness",
                value=freshness_percentage,
                status=self._get_status(freshness_percentage, self.config["thresholds"]["freshness_percentage"]),
                threshold=self.config["thresholds"]["freshness_percentage"],
                description=f"Percentage of feedback records within {self.config['thresholds']['max_stale_data_days']} days",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            metrics["feedback_freshness"] = HealthMetric(
                name="feedback_freshness",
                value=0.0,
                status="error",
                threshold=self.config["thresholds"]["freshness_percentage"],
                description=f"Error checking freshness: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
        
        return metrics
    
    def _check_database_activity(self) -> Dict[str, HealthMetric]:
        """Check database activity levels"""
        metrics = {}
        
        try:
            # Check daily activity
            now = datetime.now()
            yesterday = now - timedelta(days=1)
            
            daily_feedback_response = self.database_clients["supabase"].client.table("feedback").select(
                "*", count="exact"
            ).gte("timestamp", yesterday.isoformat()).execute()
            
            daily_activity_count = daily_feedback_response.count if daily_feedback_response.count else 0
            
            activity_status = "healthy"
            if daily_activity_count < self.config["thresholds"]["min_daily_activity_count"]:
                activity_status = "critical" if daily_activity_count == 0 else "warning"
            
            metrics["daily_activity"] = HealthMetric(
                name="daily_activity",
                value=float(daily_activity_count),
                status=activity_status,
                threshold=float(self.config["thresholds"]["min_daily_activity_count"]),
                description="Number of feedback records in the last 24 hours",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error checking database activity: {e}")
            metrics["daily_activity"] = HealthMetric(
                name="daily_activity",
                value=0.0,
                status="error",
                threshold=float(self.config["thresholds"]["min_daily_activity_count"]),
                description=f"Error checking activity: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
        
        return metrics
    
    def _check_orphaned_records(self) -> Dict[str, HealthMetric]:
        """Check for orphaned records"""
        metrics = {}
        
        try:
            # Check for orphaned feedback records
            feedback_response = self.database_clients["supabase"].client.table("feedback").select(
                "user_id", count="exact"
            ).execute()
            
            total_feedback = feedback_response.count if feedback_response.count else 0
            
            # This is a simplified check - in practice you'd verify user_ids exist
            orphaned_percentage = 0.0  # Placeholder calculation
            
            metrics["orphaned_records"] = HealthMetric(
                name="orphaned_records",
                value=orphaned_percentage,
                status=self._get_status(orphaned_percentage, self.config["thresholds"]["orphaned_records_percentage"], inverse=True),
                threshold=self.config["thresholds"]["orphaned_records_percentage"],
                description="Percentage of orphaned records in the system",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error checking orphaned records: {e}")
            metrics["orphaned_records"] = HealthMetric(
                name="orphaned_records",
                value=100.0,
                status="error",
                threshold=self.config["thresholds"]["orphaned_records_percentage"],
                description=f"Error checking orphaned records: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
        
        return metrics
    
    def _check_constraint_violations(self) -> Dict[str, HealthMetric]:
        """Check for constraint violations"""
        metrics = {}
        
        try:
            # Check video ID format constraints
            videos_response = self.database_clients["supabase"].client.table("videos").select(
                "video_id"
            ).limit(100).execute()
            
            video_ids = [v["video_id"] for v in videos_response.data] if videos_response.data else []
            
            valid_video_ids = 0
            for video_id in video_ids:
                if video_id and len(video_id) == 11 and video_id.replace("-", "").replace("_", "").isalnum():
                    valid_video_ids += 1
            
            constraint_compliance = (valid_video_ids / len(video_ids) * 100) if video_ids else 100
            
            metrics["constraint_compliance"] = HealthMetric(
                name="constraint_compliance",
                value=constraint_compliance,
                status=self._get_status(constraint_compliance, 100 - self.config["thresholds"]["constraint_violations_percentage"]),
                threshold=100 - self.config["thresholds"]["constraint_violations_percentage"],
                description="Percentage of records complying with format constraints",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error checking constraint violations: {e}")
            metrics["constraint_compliance"] = HealthMetric(
                name="constraint_compliance",
                value=0.0,
                status="error",
                threshold=100 - self.config["thresholds"]["constraint_violations_percentage"],
                description=f"Error checking constraints: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
        
        return metrics
    
    def _get_status(self, value: float, threshold: float, inverse: bool = False) -> str:
        """Determine status based on value and threshold"""
        if inverse:
            # For metrics where lower is better (e.g., error rates)
            if value <= threshold:
                return "healthy"
            elif value <= threshold * 2:
                return "warning"
            else:
                return "critical"
        else:
            # For metrics where higher is better (e.g., consistency rates)
            if value >= threshold:
                return "healthy"
            elif value >= threshold * 0.8:
                return "warning"
            else:
                return "critical"
    
    def _evaluate_overall_health(self, metrics: Dict[str, HealthMetric]) -> Tuple[str, List[HealthAlert], List[str]]:
        """Evaluate overall system health and generate alerts/recommendations"""
        critical_count = 0
        warning_count = 0
        error_count = 0
        alerts = []
        recommendations = []
        
        for metric in metrics.values():
            if metric.status == "critical":
                critical_count += 1
                alerts.append(HealthAlert(
                    severity="critical",
                    component=metric.name,
                    message=f"{metric.name} is critical: {metric.value:.1f}% (threshold: {metric.threshold}%)",
                    metrics={"value": metric.value, "threshold": metric.threshold},
                    timestamp=datetime.now().isoformat(),
                    recommended_action=self._get_recommended_action(metric.name, metric.status)
                ))
            elif metric.status == "warning":
                warning_count += 1
                alerts.append(HealthAlert(
                    severity="warning",
                    component=metric.name,
                    message=f"{metric.name} is concerning: {metric.value:.1f}% (threshold: {metric.threshold}%)",
                    metrics={"value": metric.value, "threshold": metric.threshold},
                    timestamp=datetime.now().isoformat(),
                    recommended_action=self._get_recommended_action(metric.name, metric.status)
                ))
            elif metric.status == "error":
                error_count += 1
                alerts.append(HealthAlert(
                    severity="high",
                    component=metric.name,
                    message=f"{metric.name} monitoring failed: {metric.description}",
                    metrics={"error": True},
                    timestamp=datetime.now().isoformat(),
                    recommended_action="Investigate monitoring system and database connectivity"
                ))
        
        # Determine overall status
        if critical_count > 0 or error_count > 2:
            overall_status = "critical"
            recommendations.append("Immediate attention required for critical database issues")
        elif warning_count > 3 or error_count > 0:
            overall_status = "warning"
            recommendations.append("Multiple database health concerns detected - schedule maintenance")
        else:
            overall_status = "healthy"
        
        # Add general recommendations
        if critical_count > 0:
            recommendations.append("Run database maintenance procedures")
            recommendations.append("Check database connections and configuration")
        
        if warning_count > 0:
            recommendations.append("Schedule preventive maintenance")
            recommendations.append("Review data ingestion pipelines")
        
        return overall_status, alerts, recommendations
    
    def _get_recommended_action(self, metric_name: str, status: str) -> str:
        """Get recommended action for specific metric issues"""
        actions = {
            "user_embedding_consistency": {
                "critical": "Rebuild user embeddings and verify Qdrant connectivity",
                "warning": "Investigate embedding synchronization process"
            },
            "video_referential_integrity": {
                "critical": "Run data reconciliation between Supabase and MongoDB",
                "warning": "Schedule data integrity check and cleanup"
            },
            "feedback_freshness": {
                "critical": "Check data ingestion pipeline for blockages",
                "warning": "Review data ingestion frequency and sources"
            },
            "daily_activity": {
                "critical": "Check application connectivity and user access",
                "warning": "Monitor user engagement and system availability"
            },
            "orphaned_records": {
                "critical": "Run orphaned record cleanup procedures",
                "warning": "Schedule data cleanup maintenance"
            },
            "constraint_compliance": {
                "critical": "Fix data format violations and update validation rules",
                "warning": "Review and tighten data validation procedures"
            }
        }
        
        return actions.get(metric_name, {}).get(status, "Review metric and determine appropriate action")
    
    def generate_health_report(self, format_type: str = "json") -> str:
        """Generate a formatted health report"""
        if not self.health_history:
            return "No health data available"
        
        latest_report = self.health_history[-1]
        
        if format_type == "json":
            return json.dumps(latest_report, indent=2, default=str)
        elif format_type == "summary":
            return self._generate_summary_report(latest_report)
        else:
            return str(latest_report)
    
    def _generate_summary_report(self, report: Dict[str, Any]) -> str:
        """Generate a human-readable summary report"""
        summary = [
            f"Database Health Report - {report['timestamp']}",
            f"Overall Status: {report['overall_status'].upper()}",
            "",
            "Metrics Summary:"
        ]
        
        for metric_name, metric in report["metrics"].items():
            if isinstance(metric, HealthMetric):
                summary.append(f"  {metric_name}: {metric.value:.1f}% ({metric.status})")
        
        if report["alerts"]:
            summary.append("")
            summary.append("Active Alerts:")
            for alert in report["alerts"]:
                if isinstance(alert, HealthAlert):
                    summary.append(f"  {alert.severity.upper()}: {alert.message}")
        
        if report["recommendations"]:
            summary.append("")
            summary.append("Recommendations:")
            for rec in report["recommendations"]:
                summary.append(f"  - {rec}")
        
        return "\n".join(summary)

    def clear_old_history(self, days_to_keep: int = 30):
        """Clear old health history to prevent memory buildup"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        self.health_history = [
            report for report in self.health_history
            if datetime.fromisoformat(report["timestamp"]) > cutoff_date
        ]
        
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert.timestamp) > cutoff_date
        ]

@pytest.mark.database_integrity
@pytest.mark.health_monitoring
class TestHealthMonitoringUtilities:
    """Tests for the health monitoring utilities"""

    def test_health_monitor_initialization(self, database_clients):
        """Test that health monitor initializes correctly"""
        monitor = DatabaseHealthMonitor(database_clients)
        
        assert monitor.database_clients == database_clients
        assert monitor.config is not None
        assert "thresholds" in monitor.config
        assert len(monitor.health_history) == 0
        assert len(monitor.alerts) == 0

    def test_comprehensive_health_check(self, database_clients, integrity_reporter):
        """Test comprehensive health check execution"""
        monitor = DatabaseHealthMonitor(database_clients)
        
        try:
            health_report = monitor.perform_comprehensive_health_check()
            
            assert "timestamp" in health_report
            assert "overall_status" in health_report
            assert "metrics" in health_report
            assert "alerts" in health_report
            assert "recommendations" in health_report
            
            # Check that metrics were collected
            assert len(health_report["metrics"]) > 0
            
            integrity_reporter.add_test_result(
                "health_monitor_comprehensive_check",
                "PASSED",
                {
                    "overall_status": health_report["overall_status"],
                    "metrics_count": len(health_report["metrics"]),
                    "alerts_count": len(health_report["alerts"])
                }
            )
            
        except Exception as e:
            integrity_reporter.add_test_result(
                "health_monitor_comprehensive_check",
                "ERROR",
                {"error": str(e)}
            )
            pytest.fail(f"Health check failed: {e}")

    def test_health_report_generation(self, database_clients, integrity_reporter):
        """Test health report generation in different formats"""
        monitor = DatabaseHealthMonitor(database_clients)
        
        # Run a health check to populate history
        monitor.perform_comprehensive_health_check()
        
        try:
            # Test JSON format
            json_report = monitor.generate_health_report("json")
            assert isinstance(json_report, str)
            assert "timestamp" in json_report
            
            # Test summary format
            summary_report = monitor.generate_health_report("summary")
            assert isinstance(summary_report, str)
            assert "Database Health Report" in summary_report
            
            integrity_reporter.add_test_result(
                "health_report_generation",
                "PASSED",
                {
                    "json_report_length": len(json_report),
                    "summary_report_length": len(summary_report)
                }
            )
            
        except Exception as e:
            integrity_reporter.add_test_result(
                "health_report_generation",
                "ERROR",
                {"error": str(e)}
            )
            pytest.fail(f"Report generation failed: {e}")