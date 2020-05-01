IF([Opportunity].SBQQ__Renewal__c, 

[Opportunity].Account.Name & " - "& TEXT(YEAR
([Opportunity].CloseDate )
) & " - Renewal"

, [Opportunity].Account.Name & " - " & TEXT(YEAR
(TODAY())
) & " - Templafy"

)