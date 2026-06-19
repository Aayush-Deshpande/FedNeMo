# Restructure and Fine-tune FedNeMo Presentation

The goal is to consolidate segregated content across fewer slides, address the "we are yet to build it" requirement by updating the status of the project to planned, and fine-tune the content to ensure it maximizes the chances of being shortlisted for the final round.

## User Review Required

> [!WARNING]
> Programmatically merging PowerPoint slides that contain complex diagrams (like your architecture and privacy evidence slides) can result in overlapping shapes and broken layouts. I have proposed a strategy below, but I need your feedback on how aggressively you want to condense the slides.

## Open Questions

1. **Title Slides**: The current template separates section titles onto their own slides (e.g., Slide 5 is just the words "AI Use Case"). Should I delete these title-only slides and just put the section title at the top of the content slides?
2. **AI Use Case Compression**: Your AI Use case is currently spread across 4 slides (Problem, Architecture, Privacy Evidence, Dataset/Results), each with diagrams. 
   - Option A: Keep them as 4 separate slides but remove the redundant title slide.
   - Option B: Compress all text into a single slide, which would require deleting most of the visual diagrams. 
   - Option C: Compress them into 2 slides (e.g., Problem + Architecture on one, Privacy + Dataset on another).
   Which option do you prefer?
3. **Diagrams**: Do you want to preserve the existing diagrams exactly as they are, or are you okay with me rebuilding the slides as text-heavy bullet points to fit them into fewer slides?

## Proposed Changes

### 1. Status Update (Reflecting "Yet to be built")
- **Update all status indicators**: Remove all green "BUILT (prototype)" labels and replace them with orange/grey "PLANNED" or "DESIGNED".
- **Slide 4 (Team)**: Change "Current state: working CPU prototype..." to "Planned state: federated pipeline..."
- **Slide 8 (Privacy Evidence)**: Change "■ BUILT prototype demo" to "■ PLANNED demonstration"
- **Slide 9 (Dataset)**: Change "BUILT Prototype (CPU): full federated loop runs..." to "PLANNED Prototype (CPU): full federated loop to run..."
- **Slide 12 (Bottlenecks)**: Change status from "BUILT" to "PLANNED" for the RDP accountant.
- **Slide 14 (Roadmap)**: Change "BUILT Done (prototype): Update contract..." to "PLANNED Phase 1: Update contract..."

### 2. Slide Consolidation (Pending Your Answer to Q1 & Q2)
- **Delete redundant title slides**: Remove Slide 3 (TEAM INTRODUCTION), Slide 5 (AI Use Case), Slide 10 (Tech Stack), and Slide 13 (Road Map).
- **Merge Content**: 
  - Update the remaining slides to have clear headers (e.g., "TEAM INTRODUCTION - Skillset / Experience").
  - If Option C is chosen for AI Use Case, programmatically extract the text and key points from Slides 6-9 and format them into 2 new clean slides.

## Verification Plan

### Automated Tests
- Parse the resulting `.pptx` file to ensure all references to "BUILT" are removed.
- Verify the slide count is reduced and within the 15-slide maximum.

### Manual Verification
- The user will need to open the generated `FedNeMo_Final_Updated.pptx` in PowerPoint to verify the visual layout and ensure diagrams are intact or properly summarized.
