"""HubSpot measures that retain dated observations and changes."""

CONTACT_METRICS = (
    "sessions_30d",
    "page_views_30d",
    "event_completions_30d",
    "csat_score",
    "crm_touch_count",
    "num_contacted_notes",
    "num_notes",
    "hs_email_delivered",
    "hs_email_open",
    "hs_email_click",
    "hs_email_bounce",
    "hs_email_sends_since_last_engagement",
    "hs_analytics_num_page_views",
    "hs_analytics_num_visits",
    "hs_analytics_num_event_completions",
    "hs_analytics_revenue",
    "hs_analytics_average_page_views",
    "num_conversion_events",
    "num_unique_conversion_events",
    "workspace_members_count",
    "editor_seats",
    "viewer_seats",
    "number_of_active_contracts",
    "prospecting_agent_enrolled_count",
    "hs_v2_time_in_current_stage",
    "hs_time_in_lead",
    "hs_time_in_customer",
    "touches_before_signup",
    "hs_predictivecontactscore_v2",
)
COMPANY_METRICS = (
    "customer_health_score",
    "open_support_issues",
    "csat_score",
    "csat_response_count",
    "annual_contract_value",
    "recent_deal_amount",
    "site_count",
    "editor_count",
    "num_associated_contacts",
    "num_associated_deals",
    "num_open_deals",
    "num_closed_won_deals",
    "num_contacted_notes",
    "num_notes",
)
DEAL_METRICS = (
    "amount",
    "contract_annual_value",
    "hs_deal_stage_probability",
    "hs_forecast_amount",
    "days_to_close",
    "num_contacted_notes",
    "num_notes",
    "hs_num_associated_contacts",
)
METRIC_NAMES = {
    "contacts": CONTACT_METRICS,
    "companies": COMPANY_METRICS,
    "deals": DEAL_METRICS,
}
STATE_NAMES = {
    "contacts": ("pagemantle_plan", "product_account_status", "lifecyclestage"),
    "companies": ("customer_health", "adoption_stage", "renewal_risk"),
    "deals": ("dealstage",),
}

DESCRIPTIONS = {
    "customer_health_score": "Calculated account health from the shared customer baseline and dated business events, on a 0 to 100 scale. This is not a survey response.",
    "sessions_30d": "Distinct recorded product sessions in the last 30 company-calendar days. Old sessions leave this window. Anonymous traffic is excluded.",
    "page_views_30d": "Recorded product page views in the same 30-day window. This is app use, not traffic to the marketing website.",
    "event_completions_30d": "Recorded completed product actions and approved reviews in the last 30 days. Failed actions are excluded.",
    "csat_score": "The customer's last submitted survey rating. No response means no score. A support fix alone does not change this rating.",
    "crm_touch_count": "Recorded emails, calls, meetings, notes and tasks linked to this contact. Form submissions are excluded.",
    "num_notes": "Recorded emails, calls, meetings, notes and tasks linked to this record. Despite its API name, this counts more than notes.",
    "num_contacted_notes": "Recorded emails and calls linked to this record, including inbound and outbound conversations.",
    "hs_email_delivered": "Recorded delivery confirmations for automated marketing emails. A sent email alone does not prove delivery.",
    "hs_email_open": "Recorded opens of automated marketing emails. Zero means no recorded opens, rather than proof the customer read nothing.",
    "hs_email_click": "Recorded clicks in automated marketing emails. Sales emails and product page views are excluded.",
    "hs_email_bounce": "Recorded bounces of automated marketing emails.",
    "hs_email_sends_since_last_engagement": "Automated marketing emails sent since the last recorded marketing open or click. With no recorded engagement, counts all saved sends.",
    "hs_analytics_num_page_views": "Recorded marketing website page views. Empty when no website tracking source exists.",
    "hs_analytics_num_visits": "Recorded marketing website sessions. Separate from recorded product sessions.",
    "hs_analytics_num_event_completions": "Recorded marketing website conversions. Separate from completed product actions.",
    "hs_analytics_revenue": "Revenue attributed by recorded website tracking. Empty when no revenue attribution source exists.",
    "hs_analytics_average_page_views": "Recorded website page views per recorded website session. Empty without website tracking.",
    "num_conversion_events": "Saved form submissions linked to the contact. Numeric CSAT ratings are separate observations.",
    "num_unique_conversion_events": "Distinct saved form types submitted by the contact. Numeric CSAT ratings do not add a form type.",
    "workspace_members_count": "Observed member count in the customer's product workspace. A lone snapshot does not imply older counts were the same.",
    "editor_seats": "Observed licensed editor seat count for the customer's workspace. This is not the contact's personal usage.",
    "viewer_seats": "Observed viewer seat allowance for the customer's workspace. Separate from editor seats.",
    "number_of_active_contracts": "Whether the customer's paid agreement is active at this date. Zero after its subscription ends.",
    "prospecting_agent_enrolled_count": "Number of saved prospecting-agent enrollments for this contact.",
    "hs_v2_time_in_current_stage": "Elapsed time since the contact entered its current recorded lifecycle stage, observed at the saved timestamp.",
    "hs_time_in_lead": "Total recorded time in the Lead lifecycle stage, from saved stage transitions. Empty when no dated transitions exist.",
    "hs_time_in_customer": "Total recorded time in the Customer lifecycle stage, from saved stage transitions. Separate from personal product signup.",
    "touches_before_signup": "Recorded emails, calls and meetings with this contact before their product signup date. Internal notes and tasks are excluded.",
    "hs_predictivecontactscore_v2": "Saved prediction from a contact scoring integration. Empty when no scoring observation exists. This is not calculated from CSAT.",
    "open_support_issues": "Customer-reported support cases opened and not yet confirmed resolved at this date. A workaround does not close a case.",
    "csat_response_count": "All submitted CSAT replies for this company, including repeated replies from the same person.",
    "annual_contract_value": "Annual equivalent of the agreed customer price at this date. For an inactive account, retains the last agreed price and does not mean active revenue.",
    "recent_deal_amount": "Amount of the most recently closed won deal at this date. An expansion is its incremental fee, not the whole account's annual price.",
    "site_count": "Observed product sites for this customer. Missing older observations are not a history of zero sites.",
    "editor_count": "Observed product editors for this customer, using the same people as product analytics and the warehouse.",
    "num_associated_contacts": "CRM contacts linked to this company at this date.",
    "num_associated_deals": "CRM deals linked to this company and created by this date.",
    "num_open_deals": "Linked deals whose recorded stage is still open at this date.",
    "num_closed_won_deals": "Linked deals in the recorded closed won stage at this date.",
    "amount": "Saved fee for this deal. Expansion deals use the extra annual fee, while pilots use the fee for the full pilot term.",
    "contract_annual_value": "Annual equivalent of the signed contract's full price. Can differ from an incremental expansion fee or a shorter pilot fee.",
    "hs_deal_stage_probability": "Probability assigned to the recorded deal stage. Closed won is 100 percent and closed lost is zero.",
    "hs_forecast_amount": "Saved full deal amount used for the forecast. This value is not multiplied by stage probability.",
    "days_to_close": "Days between a known deal opening and closing. Empty for imported signed contracts with no recorded negotiation start.",
    "hs_num_associated_contacts": "CRM contacts linked to this deal at this date.",
    "pagemantle_plan": "Customer plan from the paid agreement available on this date, including the last plan after cancellation.",
    "product_account_status": "Invited before this person's product signup, active during the customer's paid period, and inactive after it ends.",
    "lifecyclestage": "Recorded CRM relationship stage for this contact. Separate from product account status.",
    "customer_health": "Dated customer state based on active support cases and adoption. Red means serious trouble, yellow means concern or recovery, and green means broadly healthy.",
    "adoption_stage": "Dated rollout state. Open problems or reduced adoption can make use underused. Ended paid subscriptions are dormant.",
    "renewal_risk": "Dated risk from support, adoption and subscription state. A recent fix can reduce risk gradually. Ended paid subscriptions remain high risk.",
    "dealstage": "Saved CRM deal stage at this date. Imported signed contracts begin with the recorded closed won stage.",
}
MONEY_NAMES = {
    "annual_contract_value",
    "recent_deal_amount",
    "amount",
    "contract_annual_value",
    "hs_forecast_amount",
    "hs_analytics_revenue",
}


def activity_summary(activities, object_type):
    """Native sales-activity measures, once per linked activity, not per link."""
    from datetime import datetime

    sales = {
        row["id"]: row
        for row in activities
        if row["activity_type"] in {"email", "call", "meeting", "note", "task"}
    }
    latest = max(
        sales.values(),
        key=lambda r: (
            datetime.fromisoformat(r["occurred_at"].replace("Z", "+00:00")),
            r["id"],
        ),
        default=None,
    )
    result = dict(
        num_notes=str(len(sales)),
        num_contacted_notes=str(
            sum(r["activity_type"] in {"email", "call"} for r in sales.values())
        ),
        hs_last_sales_activity_timestamp=latest["occurred_at"] if latest else "",
    )
    result[
        "last_activity_date" if object_type == "companies" else "notes_last_updated"
    ] = (latest["occurred_at"] if latest else "")
    if object_type == "contacts":
        result["crm_touch_count"] = str(len(sales))
    return result


def metric_definitions(properties):
    """Use the dataset's labels and types, including older property catalogs."""
    return [
        {
            "objectType": item["object_type"],
            "name": item["name"],
            "label": item["label"],
            "type": item["type"],
            "description": (
                "Average of each respondent's latest submitted rating at this date, on a 1 to 5 scale. A missing reply is not zero."
                if item["object_type"] == "companies" and item["name"] == "csat_score"
                else DESCRIPTIONS[item["name"]]
            ),
            "unit": {
                "csat_score": "1 to 5",
                "customer_health_score": "0 to 100",
                "hs_deal_stage_probability": "%",
                "hs_v2_time_in_current_stage": "days",
                "hs_time_in_lead": "days",
                "hs_time_in_customer": "days",
                "days_to_close": "days",
                "hs_predictivecontactscore_v2": "score units",
                "hs_analytics_average_page_views": "page views per session",
            }.get(
                item["name"],
                (
                    "currency"
                    if item["name"] in MONEY_NAMES
                    else "count" if item["type"] == "number" else "state"
                ),
            ),
        }
        for item in properties
        if item["name"]
        in (
            METRIC_NAMES.get(item["object_type"], ())
            + STATE_NAMES.get(item["object_type"], ())
        )
    ]
