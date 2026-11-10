# AI-Based Prediction of Disruptions in Inter-City Railway Transport

**Qing He**  
Mid Sweden University

Course project 21, [AI-Based Prediction of Disruptions in Railway Transport](https://uppsala.instructure.com/courses/119802/pages/project-list?module_item_id=1662678). The project brief below is followed by the [course requirements for HT2026](#course-requirements-for-ht2026), checked on 10 November 2026.

## Motivation and project objective

**How likely is a journey to suffer a serious disruption?**

| Perspective / objective | Description |
| --- | --- |
| Passenger perspective | Estimate risk to consider an earlier train, another route or a change of plans. |
| Project objective | Predict final arrival delay and disruption severity from historical railway data and weather. |

Research focus: accuracy, the added value of weather, and how risk updates during travel.

## Example case

A simple example is:

A passenger plans to take an ICE from Frankfurt to Berlin tomorrow afternoon. The weather forecast indicates unusually high temperature. Based on historical data for similar trains, routes, times and weather conditions, can we estimate whether the journey is likely to operate normally, experience a moderate delay, suffer a major delay, or be cancelled?

## Prediction settings and scope

| Setting | Information available | Output |
| --- | --- | --- |
| Before departure | Schedule, past reliability, weather available at the prediction time | Final delay and severity probability |
| During the journey | Add current delay, earlier stops and position along the route | Updated final delay and severity probability |

**Core: one direct ICE/IC journey on one or two selected origin–destination pairs**

Extension: one connection, including missed-connection risk

### Journey and selected origin–destination pair

A **journey** is one run of one ICE or IC service on a particular date, considered between a selected origin station and destination station, without changing trains. The train must stop at the origin before reaching the destination. It may stop at other stations between them, begin before the selected origin, or continue beyond the selected destination. Intermediate stops may differ between journeys. The same train number running on another date is a different journey.

For example, a particular ICE travelling from Berlin Hbf to München Hbf on 15 October is one journey, even if it stops at Leipzig, Nürnberg and Ingolstadt. A run of the same train number on the following day is another journey.

A **corridor** means the selected origin–destination pair and the train services connecting those stations. It does not mean every track section between consecutive cities. Berlin–München is one study corridor; Berlin–Leipzig, Leipzig–Nürnberg and Nürnberg–München do not need to be counted as three separate corridors. To avoid ambiguity, use **selected origin–destination pair** from here on.

## A working definition of disruption severity

| Class | Proposed starting definition |
| --- | --- |
| Normal / minor | Final arrival delay below 20 minutes |
| Moderate | 20 to less than 60 minutes |
| Severe | 60 minutes or more |
| Cancelled | Journey cannot operate as planned |

**Initial binary target: severe delay or cancellation**

Also predict delay in minutes for completed journeys. Revisit thresholds after data exploration.

## Railway and weather datasets

| Dataset / task | Description |
| --- | --- |
| Bahn-Vorhersage | German train records since September 2021, in daily Parquet files. Reconstruct journeys from stop events. |
| DWD Climate Data Center | Historical hourly station observations: temperature, precipitation and wind. |
| Initial data task | Use a small ICE/IC subset. Join train location and time to a suitable weather station and hour. |

Data checks: access, missing records, final versus forecast updates, cancellation meaning and time zones.

## Feature groups and modelling roadmap

### Feature groups

| Feature group | Inputs |
| --- | --- |
| Service and calendar | Route, train type, time, season, duration |
| Past reliability | Earlier delay and cancellation rates |
| Weather | Temperature, rain, wind, recent extremes |
| During travel | Current delay, delay change, progress |

### Model progression

| Stage | Models |
| --- | --- |
| 0 | Historical mean / class frequency; current-delay persistence during travel |
| 1 | Linear / logistic regression |
| 2 | Random Forest; XGBoost or LightGBM |
| 3 | Optional LSTM / Transformer / GNN. Only after a stable core pipeline |

Feature ablations and model explanations help identify useful information.

## Does weather improve prediction?

**Railway features alone compared with railway features plus weather**

| Aspect | Description |
| --- | --- |
| Fair comparison | Same journeys, model and time split. Measure the change in severe-disruption detection and delay error. |
| Interpretation | A small or zero gain is a useful finding. Examine where weather helps and where it does not. |

Before departure: use past observations or forecasts issued before the prediction cutoff.

## Minimum successful project and extensions

### Minimum successful project

- A reproducible journey dataset with a clear severity target
- Baselines and standard ML models with temporal evaluation
- A fair railway-only versus railway-plus-weather experiment
- An explanation of model errors, useful features and limitations

### Optional extensions, after the core works

Dynamic risk updates, one connection, an advanced neural model, extreme-weather analysis or a simple decision-support prototype

Kickoff decisions: prediction cutoff, candidate origin–destination pairs, target and first data sample.

## Tasks for the first two weeks

- Literature review:
  - prediction targets;
  - common inputs;
  - common models;
  - evaluation methods.
- Inspect the dataset you would like to use:

  For example: the Bahn-Vorhersage dataset documentation, understand the main columns, and download only a very small sample.

  - Compare two or three possible origin–destination pairs and count the available ICE/IC journeys for each. Count each dated ICE/IC run when it stops at the origin before reaching the destination without a train change; intermediate stops may differ.
  - Select one origin–destination pair and analyse one direction first.
  - For the selected pair, examine an example journey and summarise arrival delays, cancellations and missing data.
- Investigate DWD weather data and demonstrate that they can retrieve temperature for at least one station/date and align it with a train event.
- Propose a project time plan

## Course requirements for HT2026

These requirements supplement the project brief above. Sources are the signed-in [Canvas course](https://uppsala.instructure.com/courses/119802/modules) and the university's course and syllabus pages, checked on 10 November 2026. The final presentation date is 14 January 2027, also confirmed by the project group on 10 November 2026. Unpublished details and differences between course pages are noted below.

### Course and learning outcomes

Project in Data Science, 1DL507, is a 15-credit Master's-level course. The HT2026 offering runs at 50% pace in Uppsala, on campus and in English, from 31 August 2026 to 17 January 2027. See the [official course page](https://www.uu.se/en/study/course?query=1DL507).

The [syllabus valid from autumn 2025](https://www.uu.se/en/study/syllabus?query=50869) requires students to be able to:

- Organise and complete a data-science project with a small team within its deadline.
- Break an open technical problem into tractable questions and select solutions.
- Find and assess scientific and technical sources independently.
- Produce a working solution and assess it against the specification.
- Consider the scientific, societal and ethical consequences of the work.
- Explain assumptions, justify methods and communicate results in writing and orally.

### Supervision and group work

Students work as research assistants under a supervisor. They should already understand the basic methods and seek answers independently; supervisors are not expected to teach methods from scratch. Distribute work equally and keep each student's contributions identifiable. Assessment is individual, so members of the same group may receive different grades. Supervisors can provide feedback on students' work during and after the course. See [Course Information](https://uppsala.instructure.com/courses/119802/pages/course-information?module_item_id=1734656).

At the start of the project, contact the supervisor to arrange the first meeting and send its time and location or online link to the course coordinator. Agree on subsequent meetings and project planning with the supervisor. Check that the group has access to the necessary data, software, computing resources and communication tools. The [milestone schedule](https://uppsala.instructure.com/courses/119802/pages/course-schedule-and-milestones?module_item_id=1734676) calls for regular supervisor meetings, literature review and ongoing project work.

The [introductory lecture](https://uppsala.instructure.com/courses/119802/modules/items/1742372), slides 3 and 8 to 11, names Di Yuan as coordinator/examiner and Inga Kristin Wohlert as teaching assistant. It calls for regular supervisor contact, for example at least weekly, and team meetings to discuss progress, planning and task allocation. Try to resolve technical and group-work problems within the team, record difficulties in weekly reports and seek help promptly. Contact the teachers immediately for urgent or sensitive issues; unresolved team problems should be raised with the coordinator.

### Assessment and submissions

The formal assessment consists of a written report worth 10 credits and an oral presentation worth 5 credits, with pass/fail grading. See the [syllabus](https://www.uu.se/en/study/syllabus?query=50869). Canvas also requires weekly group reports, a group contract, a mid-term presentation and the project's code/scripts as part of the course work. See [Course Information](https://uppsala.instructure.com/courses/119802/pages/course-information?module_item_id=1734656).

Submit the code/scripts, final report and final slides to their respective Studium assignments. Make one submission per group for each final deliverable. Explain how the work was divided among group members. The final slides must be a PDF. See [Course Examination Instructions](https://uppsala.instructure.com/courses/119802/pages/course-examination-instructions?module_item_id=1734796).

### Weekly reports

Submit a report every week about the preceding week's progress and the following week's plan. The named contact person should upload it to the corresponding assignment. The submission applies to the whole project group. Use the [Weekly Report.docx template](https://uppsala.instructure.com/courses/119802/files/10697605?wrap=1), or recreate it in another format while retaining the exact questions. The required content covers:

| Template field | Information to provide |
| --- | --- |
| Achievements | The work completed during the previous week. |
| Problems | Difficulties encountered and whether they were resolved. |
| Meetings | The number of group meetings, their attendees and whether the group met the supervisor. |
| Project Diary | Each member's hours worked and the activities those hours covered. |
| Plan | The next week's tasks, goals, milestones and meetings. |
| Contact person of the week | The person the course staff should contact with questions. |

See the [Weekly Report 9 assignment](https://uppsala.instructure.com/courses/119802/assignments/420835) for the questions and submission instructions. The published assignments for reports 1 through 9 show a deadline of 23:59 on their respective Mondays. Later weekly-report dates appear in the milestone schedule, but their assignments were not yet published when checked.

The [introductory lecture](https://uppsala.instructure.com/courses/119802/modules/items/1742372), slide 13, also asks reports to say whether work is ahead of, on or behind schedule. It stresses timely Monday submissions because missing reports can indicate problems within the team.

### Group contract

The group contract accompanies the first weekly report, due 21 September 2026. Its purpose is to document a discussion about how the group will work together. Course staff do not evaluate the group's organisational choices. The [TeamContract-1.docx template](https://uppsala.instructure.com/courses/119802/files/10697667?wrap=1) is optional, but all questions must be answered. They cover:

- Shared project goals and whether members' ambitions agree.
- Working arrangements, meeting frequency and location, communication channels, tools and availability conflicts.
- Responses to unequal contributions, failures to communicate, members getting stuck and the project ceasing to progress.

See the [Group Contract assignment](https://uppsala.instructure.com/courses/119802/assignments/412675).

### Final report

Write a scientific report in English in the form of a short conference paper. The required length is five pages, excluding tables, figures and references. Use a course LaTeX or Word template; LaTeX is preferred. The report will be available to other students on the course. Identify each student's text contributions in the submission comment or explain the division of work at the beginning of the report, for example in footnotes. See [Course Information](https://uppsala.instructure.com/courses/119802/pages/course-information?module_item_id=1734656) and [Templates](https://uppsala.instructure.com/courses/119802/pages/templates?module_item_id=1734699).

The course suggests the following structure:

| Section | Suggested content |
| --- | --- |
| Introduction | Application area, motivation, problem and research questions, a brief account of the work and the paper's organisation. |
| Related work | Existing approaches and how they motivate the topic and methods or relate to this project's contribution. |
| Methodology | Algorithms, models and mathematical formulation; reasons for the method choices; implementation languages, packages and frameworks. |
| Results | Dataset and exploratory analysis; experimental setup, including hardware and resources; results and evaluation. |
| Discussion | Interpretation of findings, strengths and weaknesses, limitations for the studied data and possible future experiments. |
| References | Sources cited in the report. |

Review the report together before submission, including sections written by different members. Use formal language, consistent terminology and formatting, and abbreviations that are used after introduction. Make figure legends readable and refer to every figure and table in the text. Check the capitalisation of section, table and figure references. The [introductory lecture](https://uppsala.instructure.com/courses/119802/modules/items/1742372), slides 20 to 22, warns that reports with presentation or style problems will be returned for revision before further grading.

Available resources on the [Templates page](https://uppsala.instructure.com/courses/119802/pages/templates?module_item_id=1734699):

- [LaTeX/Overleaf conference template](https://uppsala.instructure.com/courses/119802/files/10692831?wrap=1).
- [Word conference templates](https://uppsala.instructure.com/courses/119802/files/10692839?wrap=1).
- [Michael Thuné's guide to scientific reports](https://uppsala.instructure.com/courses/119802/files/10692817?wrap=1).
- [LaTeX presentation documentation](https://uppsala.instructure.com/courses/119802/files/10692808?wrap=1). PowerPoint is also permitted for presentations.

### Code and scripts

Submit the complete code/scripts produced for the project. Identify each student's code contributions in code comments or in the submission comment. If the submission is too large, a Git repository link with read access is permitted. See [Course Information](https://uppsala.instructure.com/courses/119802/pages/course-information?module_item_id=1734656) and [Course Examination Instructions](https://uppsala.instructure.com/courses/119802/pages/course-examination-instructions?module_item_id=1734796).

### Presentations

For the mid-term presentation, prepare about 10 minutes per group on the application and topic, available data, chosen methods, progress against the plan, difficulties and responses, and the plan for the remaining weeks. See [Course Information](https://uppsala.instructure.com/courses/119802/pages/course-information?module_item_id=1734656). The mid-term was scheduled for 3 November 2026 in room 101158. Students were required to attend their entire scheduled slot. The [23 October announcement](https://uppsala.instructure.com/courses/119802/discussion_topics/642285) made the published [mid-term schedule](https://uppsala.instructure.com/courses/119802/pages/mid-term-presentation-schedule?module_item_id=1734698) authoritative over the calendar sign-up and instructed groups to contact the coordinator directly about changes.

For the final presentation on 14 January 2027, cover the topic and relevance, data, relationship to earlier work, methods, results, strengths and weaknesses, and possible future work. Each student must present part of the work. The report's structure can also guide the presentation. See [Course Information](https://uppsala.instructure.com/courses/119802/pages/course-information?module_item_id=1734656).

Prepare a 14-minute final presentation plus one minute for questions/comments. A talk shorter than 11 minutes or longer than 17 minutes may receive a fail grade. Rehearse the timing; the teacher will signal at 14 minutes if the talk is still running. Submit the slides as a PDF, once per group. The group review pairings and the detailed final presentation schedule were still marked TBD when checked. See [Course Examination Instructions](https://uppsala.instructure.com/courses/119802/pages/course-examination-instructions?module_item_id=1734796) and [Final Presentation Schedule](https://uppsala.instructure.com/courses/119802/pages/final-presentation-schedule?module_item_id=1734809).

Each group must read another group's report and prepare two questions/comments to raise after that group's presentation. See the [introductory lecture](https://uppsala.instructure.com/courses/119802/modules/items/1742372), slide 16. The assigned pairing remains to be published.

### Dates and milestones

The following dates come from [Course Schedule and Milestones](https://uppsala.instructure.com/courses/119802/pages/course-schedule-and-milestones?module_item_id=1734676), supplemented by the published [assignments](https://uppsala.instructure.com/courses/119802/assignments) and announcements where noted. Dates describe course requirements, not confirmation that this group completed a submission.

| Date | Course milestone or deadline |
| --- | --- |
| 1 September 2026, 15:15 to 17:00 | Introductory lecture, Å2005. |
| 3 September 2026 | Project application deadline. |
| 8 September 2026 | Project assignments published. |
| Weeks 37 or 38 | First supervisor meeting; communicate the arrangements to the coordinator. |
| 21 September 2026, 23:59 | Weekly report 1 and group contract, confirmed by their assignments. |
| 28 September 2026, 23:59 | Weekly report 2. |
| 30 September 2026 | Follow-up meeting sign-up deadline in the 17 September announcement; see the discrepancy below. |
| Weeks 41 and 42 | Follow-up meetings about group work in room 105170; every group member must attend. |
| 5, 12, 19 and 26 October 2026, 23:59 | Weekly reports 3, 4, 5 and 6 respectively. |
| 20 October 2026 | Mid-term slot sign-up deadline in the 25 September announcement. |
| 2 November 2026, 23:59 | Weekly report 7. |
| 3 November 2026 | Mid-term presentations, Å101158. |
| 9 November 2026, 23:59 | Weekly report 8. |
| 16 November 2026, 23:59 | Weekly report 9. |
| 23 and 30 November; 7 and 14 December 2026 | Further weekly-report deadlines listed in the milestone schedule. |
| Week 51 | Report writing, final-presentation preparation and code submission if ready. |
| 8 January 2027 | Final deadline for code/scripts. |
| 11 January 2027 | Final deadline for the project report. |
| 13 January 2027 | Final deadline for the final presentation slides, in PDF format. |
| 14 January 2027 | Final presentations, Å2002 according to the milestone schedule. Date also confirmed by the project group. |

Source differences and details still awaiting publication as of 10 November 2026:

- The examination instructions still show TBD for the three final submission deadlines. The dated milestone page supplies 8, 11 and 13 January 2027, as recorded above. Final deadline times are not specified there; check the corresponding assignments when published.
- The detailed final presentation schedule and review pairings are still TBD. The presentation date is 14 January 2027; individual group slots remain unpublished.
- The milestone page prints the first weekly-report date with the year 2025. The HT2026 assignment gives Monday, 21 September 2026, at 23:59, which is the date recorded here.
- The [follow-up meeting announcement of 17 September](https://uppsala.instructure.com/courses/119802/discussion_topics/629589) gives a sign-up deadline of 30 September, whereas the milestone page lists 6 October. The [25 September mid-term announcement](https://uppsala.instructure.com/courses/119802/discussion_topics/632620) also reminds students to book the follow-up meeting. These historical sources differ; neither date is silently treated as a correction of the other.
