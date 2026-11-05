# EESM 5800 Group Project: Desktop Companion Robot

## Background

Microduck is a small bipedal robot developed by Pollen Robotics. It can walk, turn, sit, stand up, recover from a fall, and interact with objects using its beak. These functions require the integration of mechanical structures, actuators, sensors, onboard computing, power, and control software.

It provides an ideal reference for understanding how the physical structure and computational architecture of a robot jointly support its functions.

In this project, you are asked to propose a desktop companion robot on your own. You may use Microduck as a reference and study its system-level organization, but do not reproduce it. A hardware design that simply copies Microduck without changes justified by the selected companion scenario is strongly discouraged.

**Figure 1. Microduck in MuJoCo Simulation Platform**

## Objective

Propose a desktop-scale companion robot. The robot should operate safely on or near a desk and provide clearly defined companionship functions through motion, expression, communication, or physical interaction. The project emphasizes system design, robot kinematics and dynamics, human-robot interaction, and hardware-software integration.

## Project Overview

Each group shall define a desktop companionship scenario and its intended users. Possible functions include greeting, expressive movement, voice or touch response, reminders, reactions to user activities, or simple interaction with desktop objects.

Specify the user needs, operating environment, task sequence, safety constraints, and expected robot behavior. The chosen functions must be sufficiently specific for technical analysis and system design.

## Research and Concept Development

- Investigate existing desktop and companion robots.
- Study relevant mechanisms, sensors, actuators, control methods, and interaction technologies.
- Conduct a literature review and investigate user needs through observation, interviews, or questionnaires.
- Explain how your design is inspired by Microduck and how it differs from or improves upon Microduck for the selected scenario.

## Design Specifications

Provide a complete design covering:

- Dimensions, mechanical structure, degrees of freedom, joints, materials, actuators, sensors, computing hardware, communication, and power
- System architecture and information flow among sensing, computation, control, actuation, and user interaction
- Task workflow, safety measures, design constraints, engineering trade-offs, and component-selection rationale
- An implementation plan including components, algorithms, integration steps, milestones, risks, and testing methods

## Robot Kinematics and Dynamics

The report must apply relevant course knowledge to the proposed robot.

- Define coordinate frames, joint variables, and degrees of freedom, and provide suitable forward or inverse kinematics.
- Analyze relevant dynamics, such as joint forces or torques and motion constraints.
- State all assumptions and explain how the analysis affects the mechanical design, actuator selection, and control strategy.

## Human-Robot Interaction

Design a complete interaction method. Specify user inputs, robot outputs, interaction sequences, feedback, and responses to unclear or unexpected inputs. Interfaces may include speech, touch, gestures, buttons, applications, lights, sound, displays, or expressive motion. Consider usability, privacy, and safety.

## Locomotion Planning and Simulation (encouraged but not compulsory)

Groups are encouraged to develop and simulate a locomotion-planning or motion-generation algorithm, such as gait generation, trajectory planning, balance control, turning, sitting, standing, or recovery. Clearly describe the algorithm, inputs, outputs, simulation environment, and evaluation metrics.

## Prototyping and Testing (encouraged but not compulsory)

Physical prototypes, simulations, and quantitative evaluations are encouraged but not compulsory. A core function may be validated through CAD analysis, dynamic simulation, control software, an interaction prototype, sensor testing, or hardware. Report the method, metrics, results, limitations, and possible improvements.

## Deliverables

- A group report (suggested length: 10-15 pages) including:
  - Requirements and literature review
  - System architecture and design rationale
  - Kinematics and dynamics analysis
  - Human-robot interaction design
  - Hardware and software specifications
  - Implementation plan
  - Validation results, if available
- A 12-minute presentation followed by a 3-minute Q&A session.

## Submissions

- One group only needs to submit one report and PPT. Name them as: Groupx_Report and Groupx_Presentation. List all group members on the first page of the report and slides.

## Evaluation Criteria

- Innovation and creativity in design
- Completeness and rationality
- Quality of the report and presentation
- Teamwork and collaboration

## Important notes

1. Form your own team, with 7-8 students per group. If you cannot form a group by the group-formation deadline announced on Canvas, the remaining students will be randomly assigned to groups with vacancies.
2. Each group should submit one report (in PDF format) and one presentation file. State each member's contribution as a percentage based on group self-evaluation. For example: member A, 50%; member B, 20%; member C, 20%; member D, 10%. If no contribution statement is provided, all group members will receive equal marks.
3. To minimize transition time, all groups should use the classroom PC for their presentations. The presentation files will be uploaded to the classroom PC in advance.
4. The weight of the mark (Total 30%):
   - a) Report (15%)
   - b) PPT & Presentation (15%; peer review by classmates): Each group will rank all other groups. Marks will be awarded based on the average ranking; for example, 1st place receives 15 marks, 2nd place receives 14 marks, and so on.
